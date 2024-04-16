from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import PIL
import torch
from lightning.pytorch.utilities.parsing import AttributeDict
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.data._utils import download_data

__all__ = ["FastSAMDetector", "GridDetector"]


class BaseDetector(ABC):
    """Base class for detectors."""

    def __call__(self, *args, **kwargs) -> list[list[dict]]:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> list[list[dict]]:
        """Forward pass."""
        raise NotImplementedError


class FastSAMDetector(BaseDetector, YOLO):
    """FastSAM model for detection.

    Reference:
        Zhao et al. Fast Segment Anything. 2023.
    """

    model_url: str = "https://drive.google.com/uc?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv"

    def __init__(
        self,
        *args,
        model_dir: str = "models/",
        image_size: int = 1024,
        iou: float = 0.9,
        confidence: float = 0.4,
        max_det: int = 16,
        **kwargs,
    ) -> None:
        # save hyperparameters
        self.hparams = AttributeDict()
        self.hparams["model_dir"] = model_dir
        self.hparams["image_size"] = image_size
        self.hparams["iou"] = iou
        self.hparams["confidence"] = confidence
        self.hparams["max_det"] = max_det

        self.prepare_model()

        model_path = Path(self.hparams.model_dir, "FastSAM-x.pt")
        super().__init__(*args, model=model_path, **kwargs)

    def prepare_model(self) -> None:
        """Download model if necessary."""
        model_path = Path(self.hparams.model_dir, "FastSAM-x.pt")
        if model_path.exists():
            return

        download_data(self.model_url, model_path, from_gdrive=True)

    def forward(self, *args, images_fp: str | list[str], **kwargs) -> list[list[dict]]:
        """Forward pass.

        Args:
            images_fp (str | list[str]): Path to image or list of paths to images.
        """
        if isinstance(images_fp, str):
            images_fp = [images_fp]

        results = self.predict(
            source=images_fp,
            imgsz=self.hparams.image_size,
            retina_masks=True,
            iou=self.hparams.iou,
            conf=self.hparams.confidence,
            max_det=self.hparams.max_det,
            verbose=False,
        )

        results = [self._format_image_annotations(result) for result in results]
        return results

    def _format_image_annotations(self, result: Results, min_points: int = 0) -> list[dict]:
        """Format image annotations.

        Args:
            result (Results): Results from the model.
            min_points (int, optional): Minimum number of points in a mask. Defaults to 0.
        """
        annotations = []

        if result.masks is None:
            return annotations

        n = len(result.masks.data)
        for i in range(n):
            annotation = {}
            mask = result.masks.data[i] == 1.0

            if torch.sum(mask) < min_points:
                continue

            annotation["id"] = i
            annotation["segmentation"] = mask.cpu().numpy()
            annotation["bbox"] = result.boxes.data[i][:4]
            annotation["bbox"] = [int(x) for x in annotation["bbox"]]
            annotation["bbox"][2] -= annotation["bbox"][0]
            annotation["bbox"][3] -= annotation["bbox"][1]
            annotation["score"] = result.boxes.conf[i]
            annotation["area"] = annotation["segmentation"].sum()
            annotations.append(annotation)
        return annotations


class GridDetector(BaseDetector):
    """Grid detector.

    Divide images into a grid and return a list of bounding boxes for each cell.

    Args:
        grid_size (int | list[int]): Number of cells in each row and column. If a list is provided,
            runs the detector for each grid size and returns a list of lists of annotations.
            Defaults to 3.
    """

    def __init__(self, grid_size: int | list[int] = 3) -> None:
        super().__init__()
        self.grid_size = [grid_size] if isinstance(grid_size, int) else grid_size

    def forward(self, *args, images_pil: list[PIL.Image.Image], **kwargs) -> list[list[dict]]:
        """Detect grid.

        Args:
            images_pil (list[PIL.Image.Image]): List of images in PIL format.
        """
        images_annotations = []
        for image_pil in images_pil:
            image_annotations = []
            for grid_size in self.grid_size:
                width, height = image_pil.size
                annotations = self._detect(width, height, grid_size)
                image_annotations.extend(annotations)
            images_annotations.append(image_annotations)

        return images_annotations

    def _detect(self, width: int, height: int, grid_size: int) -> list[dict]:
        """Detect grid.

        Args:
            width (int): Image width.
            height (int): Image height.
        """
        grid_height, grid_width = height // grid_size, width // grid_size

        annotations = []
        for i in range(grid_size):
            for j in range(grid_size):
                annotation = {}
                annotation["id"] = i * grid_size + j
                annotation["bbox"] = [j * grid_width, i * grid_height, grid_width, grid_height]
                annotation["bbox"] = torch.tensor(annotation["bbox"])
                annotation["segmentation"] = np.zeros((height, width), dtype=np.uint8)
                start_x, start_y = j * grid_width, i * grid_height
                end_x, end_y = (j + 1) * grid_width, (i + 1) * grid_height
                annotation["segmentation"][start_y:end_y, start_x:end_x] = 1
                annotation["area"] = annotation["segmentation"].sum()
                annotations.append(annotation)

        return annotations
