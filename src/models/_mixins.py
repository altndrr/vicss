from pathlib import Path

import PIL
import torch

from src import utils
from src.models._base import BaseModel
from src.models.components.detection import BaseDetector
from src.models.components.extraction import BaseExtractor
from src.models.components.vocabularies import BaseVocabulary


class SegmentationMixin:
    """Mixin class for semantic segmentation models.

    It defines a detector and extractor to use in the model. The detector is used to detect
    objects in the images. The extractor is used to extract patches from the images. The mixin
    provides methods to extract patches from the images and to group the predictions by image.
    Moreover, it can be used to compute the semantic masks from the patches data.

    Args:
        detector (BaseDetector): Detector to detect objects in the image.
        extractor (BaseExtractor): Extractor to extract patches from the image.
    """

    def __init__(
        self,
        *args,
        detector: BaseDetector | None = None,
        extractor: BaseExtractor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.detector = detector
        self.extractor = extractor

        assert isinstance(self, BaseModel), "mixin must be used with BaseModel"

    def extract_patches(
        self, images_fp: list[str], images_pil: list[PIL.Image.Image]
    ) -> tuple[list, list, list]:
        """Extract patches from the images.

        Args:
            images_fp (list[str]): List of paths to the images.
            images_pil (list[PIL.Image.Image]): List of images in PIL format.
        """
        images_annotations = self.detector(images_fp=images_fp, images_pil=images_pil)
        images_annotations, images_patches = self.extractor(
            images_pil=images_pil, images_annotations=images_annotations
        )
        patches = sum(images_patches, [])

        patches_mask = [d["segmentation"] for det in images_annotations for d in det]
        patches_per_image = [len(image_patches) for image_patches in images_patches]

        return patches, patches_mask, patches_per_image

    def compute_semantic_masks(
        self, images_pil: list[PIL.Image.Image], images_vocab: list[list], images_data: list[tuple]
    ) -> list:
        """Compute semantic masks from the prediction grouped by image.

        Args:
            images_pil (list[PIL.Image.Image]): List of images in PIL format.
            images_vocab (list[list]): List of image vocabularies.
            images_data (list): List of image data. It contains the words, the values and
                the mask of each patch.
        """

        # wrap the function in a map_reduce decorator, so that it can be performed in parallel
        # it is not possible to use the decorator directly on the method, because it depends
        # on the number of workers, which is not known at the time of the class definition
        @utils.map_reduce(num_workers=self._num_workers, reduce="sum")  # type: ignore
        def compute(images_pil: list, images_vocab: list, images_data: list) -> list:
            """Compute semantic masks with the map_reduce decorator."""
            masks_shape = [(len(v), *i.size[::-1]) for i, v in zip(images_pil, images_vocab)]
            masks = [torch.zeros(mask_shape) for mask_shape in masks_shape]
            masks_counts = [torch.zeros((1, *mask_shape[1:])) for mask_shape in masks_shape]

            for i, (image_vocab, image_data) in enumerate(zip(images_vocab, images_data)):
                for patch_words, patch_words_values, patch_mask in image_data:
                    for word, value in zip(patch_words, patch_words_values):
                        word_idx = image_vocab.index(word)
                        masks[i][word_idx, patch_mask] = value
                    masks_counts[i][:, patch_mask] += 1

            masks = [(m / c.clamp_(min=1)).cpu().numpy() for m, c in zip(masks, masks_counts)]

            return masks

        return compute(images_pil, images_vocab, images_data)


class VocabularyFreeMixin:
    """Mixin class for vocabulary-free models.

    It defines a vocabulary method to generate a list of candidate words for each image.
    It defines a method to encode the vocabulary, and functionalities to cache the generated
    vocabularies.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        vocab_prompt (str): Prompt to use for a vocabulary. Defaults to "{}".
    """

    def __init__(
        self,
        *args,
        vocabulary: BaseVocabulary | None = None,
        vocab_prompt: str = "{}",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.vocabulary = vocabulary
        self.vocab_prompts = [vocab_prompt]
        self._prev_vocab_words = None
        self._prev_used_prompts = None
        self._prev_vocab_words_z = None

        assert isinstance(self, BaseModel), "mixin must be used with BaseModel"

    def setup(self, stage: str) -> None:
        """Setup the model.

        Args:
            stage (str): Stage of the model.
        """
        super().setup(stage)  # pytype: disable=attribute-error

        # set cache dir for vocabulary
        cache_dir = Path(".cache") / "vocabularies"
        model_name = self.hparams.model_name  # pytype: disable=attribute-error
        data_name = self.trainer.datamodule.name  # pytype: disable=attribute-error
        cache_dir = cache_dir / model_name.replace("/", "_").lower()
        cache_dir = cache_dir / self.vocabulary.name.lower()
        self.vocabulary.cache_fp = cache_dir / f"{data_name.lower()}.parquet"

    def encode_vocabulary(self, vocab: list[str], use_prompts: bool = False) -> torch.Tensor:
        """Encode a vocabulary.

        Args:
            vocab (list): List of words.
        """
        if vocab == self._prev_vocab_words and use_prompts == self._prev_used_prompts:
            return self._prev_vocab_words_z

        prompts = self.vocab_prompts if use_prompts else None
        texts = self.text_preprocess(vocab, prompts=prompts)  # pytype: disable=attribute-error
        texts_z_views = self.encode_text(texts)  # pytype: disable=attribute-error

        # cache vocabulary
        self._prev_vocab_words = vocab
        self._prev_used_prompts = use_prompts
        self._prev_vocab_words_z = texts_z_views

        return texts_z_views

    def on_test_end(self) -> None:
        """Lightning hook called at the end of the test epoch."""
        super().on_test_end()  # pytype: disable=attribute-error
        self.vocabulary.save_cache()
