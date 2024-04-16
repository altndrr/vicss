from abc import ABC
from collections.abc import Callable
from pathlib import Path

import torch
import torchvision.transforms as T
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric

from src import utils
from src.data.components.transforms import TextCompose, default_vocab_transform
from src.models._base import VisionLanguageModel
from src.models._mixins import SegmentationMixin
from src.models.components.metrics import (
    SemanticClusterAccuracy,
    SemanticJaccardIndex,
    SemanticRecall,
    SentenceIOU,
    SentenceScore,
    UniqueValues,
)
from src.models.vocabulary_free_clip import BaseVocabularyFreeCLIP

log = utils.get_logger(__name__)


class CaSED:
    """CaSED model factory.

    Args:
        task (str): Task to perform.
    """

    def __new__(cls, task: str, *args, **kwargs) -> VisionLanguageModel:
        if task == "classification":
            return ClassificationCaSED(*args, **kwargs)
        elif task == "segmentation":
            return SegmentationCaSED(*args, **kwargs)
        raise ValueError(f"Invalid task {task}")


class BaseCaSED(BaseVocabularyFreeCLIP, ABC):
    """LightningModule for Category Search from External Databases.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        vocab_transform (TextCompose, optional): List of transforms to apply to a vocabulary.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        alpha (float): Weight for the average of the image and text predictions. Defaults to 0.5.
        vocab_prompt (str): Prompt to use for a vocabulary. Defaults to "{}".
        vocab_prompts_from_dataset (bool): Whether to use vocabulary prompts from the dataset.
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    task: str

    def __init__(
        self,
        *args,
        vocab_prompts_from_dataset: bool = False,
        vocab_transform: TextCompose | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._vocab_transform = vocab_transform or default_vocab_transform()
        self._vocab_prompts_from_dataset = vocab_prompts_from_dataset

        # save hyperparameters
        kwargs["alpha"] = kwargs.get("alpha", 0.5)
        self.save_hyperparameters("alpha", "vocab_prompts_from_dataset", "vocab_transform")

    def setup(self, stage: str) -> None:
        """Setup the model.

        Args:
            stage (str): Stage of the model.
        """
        super().setup(stage)

        # if needed, load vocabulary prompts from the dataset
        if self._vocab_prompts_from_dataset:
            artifact_dir = Path(self.trainer.datamodule.hparams.artifact_dir)
            datamodule_name = self.trainer.datamodule.alt_name
            prompts_fp = Path(artifact_dir) / "data" / datamodule_name / "prompts.txt"
            log.info(f"Loading vocabulary prompts from {prompts_fp}")
            self.vocab_prompts = open(prompts_fp).read().splitlines()

    @property
    def vocab_transform(self) -> Callable:
        """Get image preprocess transform.

        The getter wraps the transform in a map_reduce function and applies it to a list of images.
        If interested in the transform itself, use `self._vocab_transform`.
        """
        vocab_transform = self._vocab_transform

        def vocabs_transforms(texts: list[str]) -> list[torch.Tensor]:
            return [vocab_transform(text) for text in texts]

        return vocabs_transforms

    @vocab_transform.setter
    def vocab_transform(self, transform: T.Compose) -> None:
        """Set image preprocess transform.

        Args:
            transform (torch.nn.Module): Transform to use.
        """
        self._vocab_transform = transform

    def batch_step(
        self, images_z: torch.Tensor, vocabularies: list[list]
    ) -> tuple[torch.Tensor, list, list]:
        """Perform a single batch step.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
            images_fp (list[str]): List of paths to image files.
        """
        unfiltered_words = sum(vocabularies, [])

        # encode unfiltered words
        unfiltered_words_z = self.encode_vocabulary(unfiltered_words).squeeze(0)
        unfiltered_words_z = unfiltered_words_z / unfiltered_words_z.norm(dim=-1, keepdim=True)

        # generate a text embedding for each image from their unfiltered words
        unfiltered_words_per_image = [len(vocab) for vocab in vocabularies]
        texts_z = torch.split(unfiltered_words_z, unfiltered_words_per_image)
        texts_z = torch.stack([word_z.mean(dim=0) for word_z in texts_z])
        texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

        # filter the words and embed them
        vocabularies = self.vocab_transform(vocabularies)
        vocabularies = [vocab or ["object"] for vocab in vocabularies]
        words = sum(vocabularies, [])
        words_z = self.encode_vocabulary(words, use_prompts=True)
        words_z = words_z / words_z.norm(dim=-1, keepdim=True)

        # create a one-hot relation mask between images and words
        words_per_image = [len(vocab) for vocab in vocabularies]
        col_indices = torch.arange(sum(words_per_image))
        row_indices = torch.arange(len(images_z)).repeat_interleave(torch.tensor(words_per_image))
        mask = torch.zeros(len(images_z), sum(words_per_image), device=self.device)
        mask[row_indices, col_indices] = 1

        # get the image and text predictions
        images_p = self.classifier(images_z, words_z, mask=mask)
        texts_p = self.classifier(texts_z, words_z, mask=mask)

        # average the image and text predictions
        samples_p = self.hparams.alpha * images_p + (1 - self.hparams.alpha) * texts_p

        return samples_p, words, vocabularies


class ClassificationCaSED(BaseCaSED):
    """LightningModule for Category Search from External Databases.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        vocab_transform (TextCompose, optional): List of transforms to apply to a vocabulary.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        alpha (float): Weight for the average of the image and text predictions. Defaults to 0.5.
        vocab_prompt (str): Prompt to use for the vocabulary. Defaults to "{}".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    task: str = "classification"

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Lightning test step.

        Args:
            batch (dict): Batch of data.
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


class SegmentationCaSED(SegmentationMixin, BaseCaSED):
    """LightningModule for Category Search from External Databases.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        vocab_transform (TextCompose, optional): List of transforms to apply to the vocabulary.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        alpha (float): Weight for the average of the image and text predictions. Defaults to 0.5.
        detector (BaseDetector): Detector to detect objects in the image.
        extractor (BaseExtractor): Extractor to extract patches from the image.
        vocab_prompt (str): Prompt to use for the vocabulary. Defaults to "{}".
        vocab_source (str): Source to use for the vocabulary. Either "patch" or "image". Defaults
            to "patch".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    task: str = "segmentation"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # save hyperparameters
        kwargs["vocab_source"] = kwargs.get("vocab_source", "patch")
        assert kwargs["vocab_source"] in ["patch", "image"], "Invalid `vocab_source`"
        self.save_hyperparameters("vocab_source")

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
            batch (dict): Batch of data.
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
        if self.hparams.vocab_source == "patch":
            patches_vocab = self.vocabulary(images_z=patches_z)
        elif self.hparams.vocab_source == "image":
            images = batch["images_tensor"]
            images_z = self.vision_encoder(images)
            images_vocab = self.vocabulary(images_z=images_z, images_fp=images_fp)

            # expand image vocabularies to patches
            patches_vocab = []
            for i, patch_per_image in enumerate(patches_per_image):
                for _ in range(patch_per_image):
                    patches_vocab.append(images_vocab[i])

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
    _ = CaSED(task="classification")
