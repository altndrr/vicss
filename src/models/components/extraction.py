from abc import ABC, abstractmethod

import cv2
import numpy as np
import PIL
import torch

__all__ = ["PatchExtractor"]


_EXTRACTOR_OUTPUT = tuple[list[list[dict]], list[list[PIL.Image.Image]]]


class BaseExtractor(ABC):
    """Base class for extractors."""

    def __call__(self, *args, **kwargs) -> _EXTRACTOR_OUTPUT:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> _EXTRACTOR_OUTPUT:
        """Forward pass."""
        raise NotImplementedError


class PatchExtractor(BaseExtractor):
    """Patch extractor.

    Extract patches from images given annotations.

    Args:
        enlarge_bbox (float, optional): Enlarge bbox. Defaults to 0.1.
        extraction_method (str, optional): Extraction method. Defaults to "crop".
        min_segmentation_area (int, optional): Minimum segmentation area. Defaults to 1000.
        retain_full_image (bool, optional): Return also full image as patch. Defaults to False.
    """

    def __init__(
        self,
        enlarge_bbox: float = 0.0,
        extraction_method: str = "crop",
        min_segmentation_area: int = 1000,
        retain_full_image: bool = False,
    ) -> None:
        assert extraction_method in ["crop", "mark"]

        self.extraction_method = extraction_method
        self.min_segmentation_area = min_segmentation_area
        self.enlarge_bbox = enlarge_bbox
        self.retain_full_image = retain_full_image

    def forward(
        self, *args, images_pil: list[PIL.Image.Image], images_annotations: list[dict], **kwargs
    ) -> _EXTRACTOR_OUTPUT:
        """Extract patches from images given annotations.

        Args:
            images_pil (list[PIL.Image.Image]): List of images in PIL format.
            images_annotations (list[dict]): List of image annotations.
        """
        images_patches, used_images_annotations = [], []
        for image_pil, image_annotations in zip(images_pil, images_annotations):
            used_image_annotations, image_patches = self._extract(image_pil, image_annotations)
            used_images_annotations.append(used_image_annotations)
            images_patches.append(image_patches)

        return used_images_annotations, images_patches

    def _extract(
        self, image_pil: PIL.Image.Image, image_annotations: dict
    ) -> tuple[list[dict], list[PIL.Image.Image]]:
        """Extract patches from a  single image.

        Args:
            image_pil (PIL.Image.Image): Image in PIL format.
            image_annotations (dict): Image annotations.
        """
        full_image_bbox = torch.tensor([0, 0, image_pil.width, image_pil.height])
        full_image_annotation = {
            "bbox": full_image_bbox,
            "segmentation": np.ones((image_pil.width, image_pil.height)),
        }

        # if no annotations, return full image
        if len(image_annotations) == 0:
            return [full_image_annotation], [image_pil]

        # sort annotations by segmentation area
        image_annotations = sorted(
            image_annotations, key=lambda x: np.sum(x["area"]), reverse=True
        )

        image_patches, used_image_annotations = [], []
        image_np = np.array(image_pil)
        for image_annotation in image_annotations:
            if image_annotation["area"] <= self.min_segmentation_area:
                continue

            # get bbox
            x, y, w, h = image_annotation["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h

            # enlarge bbox
            x1 = max(0, x1 - int(self.enlarge_bbox * w))
            y1 = max(0, y1 - int(self.enlarge_bbox * h))
            x2 = min(image_pil.width, x2 + int(self.enlarge_bbox * w))
            y2 = min(image_pil.height, y2 + int(self.enlarge_bbox * h))

            # extract patches
            if self.extraction_method == "crop":
                image_patch = image_np[y1:y2, x1:x2]
            elif self.extraction_method == "mark":
                image_patch = image_np.copy()
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                cv2.ellipse(image_patch, center, axes, 0, 0, 360, (255, 0, 0), 5)
            image_patch = PIL.Image.fromarray(image_patch)
            image_patches.append(image_patch)
            used_image_annotations.append(image_annotation)

        # add full image
        if self.retain_full_image:
            image_patches.append(image_pil)
            used_image_annotations.append(full_image_annotation)

        return used_image_annotations, image_patches
