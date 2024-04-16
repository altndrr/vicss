from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric

from src import utils
from src.data.components.transforms import (
    TextCompose,
    ToMultiCropTensors,
    ToRGBTensor,
    default_vocab_transform,
)
from src.models._base import VisionLanguageModel
from src.models.components.metrics import SemanticJaccardIndex, SemanticRecall, UniqueValues
from src.models.vocabulary_free_clip import BaseVocabularyFreeCLIP

log = utils.get_logger(__name__)


class DenseCaSED:
    """Dense CaSED model factory.

    Args:
        task (str): Task to perform.
    """

    def __new__(cls, task: str, *args, **kwargs) -> VisionLanguageModel:
        if task == "segmentation":
            return SegmentationDenseCaSED(*args, **kwargs)
        raise ValueError(f"Invalid task {task}")


class SegmentationDenseCaSED(BaseVocabularyFreeCLIP):
    """LightningModule for Dense Category Search from External Databases.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        vocab_transform (TextCompose, optional): List of transforms to apply to the vocabulary.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        alpha (float): Weight for the average of the image and text predictions. Defaults to 0.5.
        crop_grid_sizes (int): Grid sizes to use for multi-crop. Defaults to [2, 4, 8].
        crop_stride_grid (bool): Whether to also extract crops with a stride of 0.5. Defaults
            to False.
        crop_output_size (int): Output size of the crops. Defaults to 128.
        pixel_map_size (int | tuple[int, int]): Size of the pixel map to use for aggregating crops.
        vocab_prompt (str): Prompt to use for the vocabulary. Defaults to "{}".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    task: str = "segmentation"

    def __init__(self, *args, vocab_transform: TextCompose | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vocab_transform = vocab_transform or default_vocab_transform()

        # save hyperparameters
        kwargs["alpha"] = kwargs.get("alpha", 0.5)
        kwargs["crop_grid_sizes"] = kwargs.get("crop_grid_sizes", [2, 4, 8])
        kwargs["crop_stride_grid"] = kwargs.get("crop_stride_grid", False)
        kwargs["crop_output_size"] = kwargs.get("crop_output_size", 128)
        kwargs["pixel_map_size"] = kwargs.get("pixel_map_size", (16, 16))
        if isinstance(kwargs["pixel_map_size"], int):
            kwargs["pixel_map_size"] = (kwargs["pixel_map_size"], kwargs["pixel_map_size"])
        self.save_hyperparameters(
            "alpha",
            "crop_grid_sizes",
            "crop_stride_grid",
            "crop_output_size",
            "pixel_map_size",
            "vocab_transform",
        )

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

    def setup(self, stage: str) -> None:
        """Setup the model.

        Args:
            stage (str): Stage of the model.
        """
        super().setup(stage)

        transforms = []
        norm_mean = [0.48145466, 0.4578275, 0.40821073]
        norm_std = [0.26862954, 0.26130258, 0.27577711]
        transforms.append(ToRGBTensor())
        transforms.append(T.Normalize(mean=norm_mean, std=norm_std))
        transforms.append(
            ToMultiCropTensors(
                grid_sizes=self.hparams.crop_grid_sizes,
                stride_grid=self.hparams.crop_stride_grid,
                output_size=self.hparams.crop_output_size,
            )
        )
        self.image_preprocess = T.Compose(transforms)

        # create bbox rescale factor
        UP_H, UP_W = self.hparams.pixel_map_size
        self._bbox_rescale = torch.tensor([UP_W, UP_H, UP_W, UP_H], device=self.device)

        # setup cache system for the vocabulary
        datamodule_name = self.trainer.datamodule.name.lower()
        self._cache_dir = Path(".cache", "words", datamodule_name)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        assert self._batch_size == 1, "Batch size must be 1 for DenseCaSED"

    def batch_step(
        self, patches_z: torch.Tensor, batch_idx: int | None = None
    ) -> tuple[torch.Tensor, list, list]:
        """Perform a single batch step.

        Args:
            patches_z (torch.Tensor): Batch of patches representations.
            batch_idx (int, optional): Index of the batch. Defaults to None.
        """
        patches_per_image = [len(patches) for patches in patches_z]
        patches_z = patches_z.reshape(-1, patches_z.shape[-1])
        patches_z = patches_z / patches_z.norm(dim=-1, keepdim=True)

        # get vocabularies for each patch
        vocabularies = self.vocabulary(images_z=patches_z)
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
        words_z = self.encode_vocabulary(words, use_prompts=True).squeeze(0)
        words_z = words_z / words_z.norm(dim=-1, keepdim=True)

        # create a one-hot relation mask between images and words
        words_per_image = [len(vocab) for vocab in vocabularies]
        col_indices = torch.arange(sum(words_per_image))
        row_indices = torch.arange(len(patches_z)).repeat_interleave(torch.tensor(words_per_image))
        mask = torch.zeros(len(patches_z), sum(words_per_image), device=self.device)
        mask[row_indices, col_indices] = 1

        # get the image and text predictions
        images_p = self.classifier(patches_z, words_z, mask=mask)
        texts_p = self.classifier(texts_z, words_z, mask=mask)

        # average the image and text predictions
        samples_p = self.hparams.alpha * images_p + (1 - self.hparams.alpha) * texts_p

        preds = samples_p.topk(k=1, dim=-1)
        patches_vocab = [[words[idx] for idx in indices] for indices in preds.indices]
        images_vocab = utils.group(patches_vocab, patches_per_image)
        images_vocab = [list(sum(image_vocab, [])) for image_vocab in images_vocab]

        return samples_p, words, images_vocab

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Lightning test step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        targets = batch["masks_np"]

        images_crops = batch["images_crops_tensor"]
        images_crops_bbox = batch["images_crops_bbox"]

        # get crops representations
        UP_H, UP_W = self.hparams.pixel_map_size
        B, N, C, H, W = images_crops.shape
        images_crops_z = self.vision_encoder(images_crops.view((-1, C, H, W))).view(B, N, -1)
        images_crops_z = images_crops_z / images_crops_z.norm(dim=-1, keepdim=True)

        # rescale bboxes to original image size
        images_crops_bbox = (images_crops_bbox * self._bbox_rescale).int()

        # aggregate crops representations to compute images pixel representations
        images_pixels_z = self.accumulate_crops_z(images_crops_z, images_crops_bbox)

        # get predictions for each image
        _, _, images_vocab = self.batch_step(
            images_pixels_z.view(B, -1, images_pixels_z.shape[-1]), batch_idx=batch_idx
        )

        # compute semantic masks
        assignments = [[images_vocab[b].index(word) for word in images_vocab[b]] for b in range(B)]
        assignments = torch.tensor(assignments, device=self.device).unsqueeze(0)
        assignments = assignments.view(B, UP_H, UP_W)
        # ? by converting the assignments into a one-hot fp tensor, we basically invert
        # ? the argmax operation, and we can pass it to the metrics
        masks = F.one_hot(assignments).float().permute(0, 3, 1, 2).cpu().numpy()
        words = list(zip(images_vocab, masks))

        # log metrics
        num_vocabs = torch.tensor([len(set(image_vocab)) for image_vocab in images_vocab])
        num_vocabs = num_vocabs.to(self.device)
        self.metrics["test/num_vocabs_avg"](num_vocabs)
        self.log("test/num_vocabs.avg", self.metrics["test/num_vocabs_avg"])
        self.metrics["test/vocabs_unique"](images_vocab)
        self.log("test/vocabs.unique", self.metrics["test/vocabs_unique"])
        self.metrics["test/vocabs/selected_unique"]([w for w, _ in words])
        self.log("test/vocabs/selected.unique", self.metrics["test/vocabs/selected_unique"])
        self.metrics["test/semantic_metrics"](words, targets)
        self.log_dict(self.metrics["test/semantic_metrics"])

    def accumulate_crops_z(self, crops_z: torch.Tensor, crops_bbox: torch.Tensor) -> torch.Tensor:
        """Accumulate crops representations to compute pixel representations.

        Args:
            crops_z (torch.Tensor): Representations of the crops.
            crops_bbox (torch.Tensor): Bounding boxes of the crops.
        """
        B, _, _ = crops_z.shape
        UP_H, UP_W = self.hparams.pixel_map_size
        images_pixels_z = torch.zeros(B, UP_H, UP_W, crops_z.shape[-1], device=self.device)
        images_pixels_counts = torch.zeros(B, UP_H, UP_W, device=self.device)
        for i, (crops_z, crops_bbox) in enumerate(zip(crops_z, crops_bbox)):
            for crop_z, crop_bbox in zip(crops_z, crops_bbox):
                x1, y1, w, h = crop_bbox
                x2, y2 = x1 + w, y1 + h
                images_pixels_z[i, y1:y2, x1:x2] += crop_z
                images_pixels_counts[i, y1:y2, x1:x2] += 1
        images_pixels_z = images_pixels_z / images_pixels_counts.clamp_(min=1).unsqueeze(-1)
        images_pixels_z = images_pixels_z / images_pixels_z.norm(dim=-1, keepdim=True)

        return images_pixels_z

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
    _ = DenseCaSED(task="segmentation")
