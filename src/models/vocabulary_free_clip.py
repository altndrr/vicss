from abc import ABC

import torch
import torchvision.transforms as T
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric

from src import utils
from src.models._base import VisionLanguageModel
from src.models._mixins import SegmentationMixin, VocabularyFreeMixin
from src.models.clip import BaseCLIP
from src.models.components.metrics import (
    SemanticClusterAccuracy,
    SemanticJaccardIndex,
    SemanticRecall,
    SentenceIOU,
    SentenceScore,
    UniqueValues,
)

log = utils.get_logger(__name__)


class VocabularyFreeCLIP:
    """Vocabulary-free CLIP model factory.

    Args:
        task (str): Task to perform.
    """

    def __new__(cls, task: str, *args, **kwargs) -> VisionLanguageModel:
        if task == "classification":
            return ClassificationVocabularyFreeCLIP(*args, **kwargs)
        elif task == "segmentation":
            return SegmentationVocabularyFreeCLIP(*args, **kwargs)
        raise ValueError(f"Invalid task {task}")


class BaseVocabularyFreeCLIP(VocabularyFreeMixin, BaseCLIP, ABC):
    """LightningModule for Contrastive Language-Image Pre-training without a vocabulary.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        vocab_prompt (str): Prompt to use for a vocabulary. Defaults to "{}".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    task: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, prompt="{}", **kwargs)

    def batch_step(
        self, images_z: torch.Tensor, vocabularies: list[list]
    ) -> tuple[torch.Tensor, list, list]:
        """Perform a single batch step.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
            images_fp (list[str]): List of paths to image files.
        """
        words = sum(vocabularies, [])

        # encode vocabularies
        words_z = self.encode_vocabulary(words, use_prompts=True).squeeze(0)
        words_z = words_z / words_z.norm(dim=-1, keepdim=True)

        # create a one-hot relation mask between images and words
        words_per_image = [len(vocab) for vocab in vocabularies]
        col_indices = torch.arange(sum(words_per_image))
        row_indices = torch.arange(len(images_z)).repeat_interleave(torch.tensor(words_per_image))
        mask = torch.zeros(len(images_z), sum(words_per_image), device=self.device)
        mask[row_indices, col_indices] = 1

        images_p = self.classifier(images_z, words_z, mask=mask)

        return images_p, words, vocabularies


class ClassificationVocabularyFreeCLIP(BaseVocabularyFreeCLIP):
    """LightningModule for Contrastive Language-Image Pre-training without a vocabulary.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        vocab_prompt (str): Prompt to use for a vocabulary. Defaults to "{}".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    task: str = "classification"

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Lightning test step.

        Args:
            batch (Any): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        images = batch["images_tensor"]
        targets = batch["targets_name"]
        images_fp = batch["images_fp"]

        # get vocabularies for each image
        images_z = self.vision_encoder(images)
        images_vocab = self.vocabulary(images_z=images_z, images_fp=images_fp)

        # get predictions for each image
        images_p, words, images_vocab = self.batch_step(images_z, images_vocab)
        preds = images_p.topk(k=1, dim=-1)
        images_words = [[words[idx] for idx in indices.tolist()] for indices in preds.indices]
        images_words_values = preds.values.tolist()
        words = [
            {word: sum([v for w, v in zip(iw, iwv) if w == word]) for word in set(iw)}
            for iw, iwv in zip(images_words, images_words_values)
        ]

        # log metrics
        num_vocabs = torch.tensor([len(image_vocab) for image_vocab in images_vocab])
        num_vocabs = num_vocabs.to(self.device)
        self.metrics["test/num_vocabs_avg"](num_vocabs)
        self.log("test/num_vocabs.avg", self.metrics["test/num_vocabs_avg"])
        self.metrics["test/vocabs_unique"](images_vocab)
        self.log("test/vocabs.unique", self.metrics["test/vocabs_unique"])
        self.metrics["test/vocabs/selected_unique"](sum([list(w.keys()) for w in words], []))
        self.log("test/vocabs/selected.unique", self.metrics["test/vocabs/selected_unique"])
        self.metrics["test/semantic_iou"](words, targets)
        self.log("test/semantic_iou", self.metrics["test/semantic_iou"])
        self.metrics["test/semantic_similarity"](words, targets)
        self.log("test/semantic_similarity", self.metrics["test/semantic_similarity"])

        self.test_outputs.append((words, targets))

    def on_test_epoch_end(self) -> None:
        """Lightning hook called at the end of the test epoch."""
        words, targets = zip(*self.test_outputs)
        words = sum(words, [])
        targets = sum(targets, [])
        self.metrics["test/semantic_cluster_acc"](words, targets)
        self.log("test/semantic_cluster_acc", self.metrics["test/semantic_cluster_acc"])

        super().on_test_epoch_end()

    def configure_metrics(self) -> None:
        """Configure metrics."""
        self.metrics["test/num_vocabs_avg"] = MeanMetric()
        self.metrics["test/vocabs_unique"] = UniqueValues()
        self.metrics["test/vocabs/selected_unique"] = UniqueValues()

        semantic_cluster_acc = SemanticClusterAccuracy(task="multiclass", average="micro")
        self.metrics["test/semantic_cluster_acc"] = semantic_cluster_acc
        self.metrics["test/semantic_iou"] = SentenceIOU()
        self.metrics["test/semantic_similarity"] = SentenceScore()


class SegmentationVocabularyFreeCLIP(SegmentationMixin, BaseVocabularyFreeCLIP):
    """LightningModule for Contrastive Language-Image Pre-training without a vocabulary.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        detector (BaseDetector): Detector to detect objects in the image.
        extractor (BaseExtractor): Extractor to extract patches from the image.
        vocab_prompt (str): Prompt to use for a vocabulary. Defaults to "{}".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    task: str = "segmentation"

    def setup(self, stage: str) -> None:
        """Setup the model.

        Args:
            stage (str): Stage of the model.
        """
        super().setup(stage)

        assert self.detector is not None, "Detector must be provided"
        assert self.extractor is not None, "Extractor must be provided"

        image_preprocess = self._image_preprocess
        image_preprocess.transforms[0] = T.Resize(
            size=(224, 224),
            interpolation=T.InterpolationMode.BICUBIC,
            max_size=None,
            antialias="warn",
        )
        del image_preprocess.transforms[1]
        self.image_preprocess = image_preprocess

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Lightning test step.

        Args:
            batch (Any): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        images_pil = batch["images_pil"]
        targets = batch["masks_np"]
        images_fp = batch["images_fp"]

        # get patches from the detector
        patches_pil, patches_mask, patches_per_image = self.extract_patches(images_fp, images_pil)
        patches = torch.stack(self.image_preprocess(patches_pil)).to(self.device)

        # get vocabularies for each patch
        patches_z = self.vision_encoder(patches)
        patches_vocab = self.vocabulary(images_z=patches_z, images_pil=patches_pil)

        # get predictions for each patch
        patches_p, words, patches_vocab = self.batch_step(patches_z, patches_vocab)
        preds = patches_p.topk(k=1, dim=-1)
        patches_words = [[words[idx] for idx in indices.tolist()] for indices in preds.indices]
        patches_words_values = preds.values.tolist()

        # group vocabularies by image and remove duplicates
        images_vocab = utils.group(patches_vocab, patches_per_image)
        images_vocab = [list(set(sum(image_vocab, []))) for image_vocab in images_vocab]
        images_vocab = [vocab or ["object"] for vocab in images_vocab]

        # compute semantic masks
        patches_data = list(zip(patches_words, patches_words_values, patches_mask))
        images_data = utils.group(patches_data, patches_per_image)
        masks = self.compute_semantic_masks(images_pil, images_vocab, images_data)
        words = list(zip(images_vocab, masks))

        # log metrics
        num_vocabs = torch.tensor([len(image_vocab) for image_vocab in images_vocab])
        num_vocabs = num_vocabs.to(self.device)
        self.metrics["test/num_vocabs_avg"](num_vocabs)
        self.log("test/num_vocabs.avg", self.metrics["test/num_vocabs_avg"])
        self.metrics["test/vocabs_unique"](images_vocab)
        self.log("test/vocabs.unique", self.metrics["test/vocabs_unique"])
        self.metrics["test/vocabs/selected_unique"]([w for w, _ in words])
        self.log("test/vocabs/selected.unique", self.metrics["test/vocabs/selected_unique"])
        self.metrics["test/semantic_metrics"](words, targets)
        self.log_dict(self.metrics["test/semantic_metrics"])

    def configure_metrics(self) -> None:
        """Configure metrics."""
        self.metrics["test/num_vocabs_avg"] = MeanMetric()
        self.metrics["test/vocabs_unique"] = UniqueValues()
        self.metrics["test/vocabs/selected_unique"] = UniqueValues()

        semantic_metrics = {}
        classes = self.trainer.datamodule.classes
        sem_kwargs = {"classes": classes, "average": "macro"}
        semantic_metrics = {
            "test/semantic_jaccard_index/hard": SemanticJaccardIndex("hard", **sem_kwargs),
            "test/semantic_jaccard_index/soft": SemanticJaccardIndex("soft", **sem_kwargs),
            "test/semantic_jaccard_index/overlap": SemanticJaccardIndex("overlap", **sem_kwargs),
            "test/semantic_jaccard_index/nearest": SemanticJaccardIndex("nearest", **sem_kwargs),
            "test/semantic_recall/hard": SemanticRecall("hard", **sem_kwargs),
            "test/semantic_recall/soft": SemanticRecall("soft", **sem_kwargs),
        }
        self.metrics["test/semantic_metrics"] = MetricCollection(semantic_metrics)


if __name__ == "__main__":
    _ = VocabularyFreeCLIP(task="classification")
