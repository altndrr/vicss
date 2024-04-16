from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset

__all__ = ["ClassificationDataset", "SegmentationDataset"]


class ClassificationDataset(VisionDataset):
    """Dataset for image classification.

    If only the root directory is provided, the dataset works as the `ImageFolder` dataset from
    torchvision. It is otherwise possible to provide a list of images and/or labels. To modify
    the class names, it is possible to provide a list of class names. If the class names are not
    provided, they are inferred from the folder names.

    Args:
        root (str): Root directory of dataset where `images` are found.
        images (list[str], optional): List of images. Defaults to None.
        labels (list[int] | list[list[int]], optional): List of labels (supports multi-labels).
            Defaults to None.
        class_names (list[str], optional): List of class names. Defaults to None.
        classes_to_idx (list[str], optional): Mapping from class names to class indices. Defaults
            to None.
        transform (Callable | list[Callable], optional): A function/transform that takes in a
            PIL image and returns a transformed version. If a list of transforms is provided, they
            are applied depending on the target label. Defaults to None.
        target_transform (Callable, optional): A function/transform that takes in the target and
            transforms it. Defaults to None.

     Attributes:
        class_names (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index, domain_index) tuples.
        images (list): List of paths to images.
        targets (list): The class_index value for each image in the dataset.
    """

    def __init__(
        self,
        root: str,
        images: list[str] | None = None,
        labels: list[int] | list[list[int]] | None = None,
        class_names: list[str] | None = None,
        classes_to_idx: dict[str, int] | None = None,
        transform: Callable | list[Callable] | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        if not images:
            images = [str(path) for path in Path(root).glob("*/*")]

        if not class_names:
            class_names = {Path(f).parent.name for f in images}

        if not labels:
            folder_names = {Path(f).parent.name for f in images}
            folder_names = sorted(folder_names)
            folder_names_to_idx = {c: i for i, c in enumerate(folder_names)}
            labels = [folder_names_to_idx[Path(f).parent.name] for f in images]

        self.samples = list(zip(images, labels))
        self.images = images
        self.targets = labels
        self.is_multi_label = all(isinstance(t, list) for t in labels)

        self.class_names = class_names
        self.classes_to_idx = classes_to_idx or {c: i for i, c in enumerate(self.class_names)}

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.loader = default_loader

    def __getitem__(self, index: int) -> dict:
        path, targets_idx = self.samples[index]
        targets_idx = [targets_idx] if isinstance(targets_idx, int) else targets_idx
        targets_name = [self.class_names[t] for t in targets_idx]

        targets = torch.tensor(targets_idx, dtype=torch.long)
        image_pil = self.loader(path)
        if self.transform is not None:
            if isinstance(self.transform, list):
                image_tensor = self.transform[targets](image_pil)
            else:
                image_tensor = self.transform(image_pil)
        if self.target_transform is not None:
            targets = [self.target_transform(t) for t in targets]

        targets_one_hot = torch.zeros(len(self.class_names), dtype=torch.long)
        targets_one_hot[targets] = 1

        data = {
            "images_fp": path,
            "images_pil": image_pil,
            "images_tensor": image_tensor,
            "targets_idx": targets_idx,
            "targets_one_hot": targets_one_hot,
            "targets_name": targets_name,
        }
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        if self.class_names is not None:
            if len(self.class_names) > 10:
                body += [f"Classes: {', '.join(self.class_names[:10])}..."]
            else:
                body += [f"Classes: {', '.join(self.class_names)}"]
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


class SegmentationDataset(VisionDataset):
    """Dataset for semantic segmentation.

    Args:
        root (str): Root directory of dataset where `images` are found.
        images (list[str], optional): List of images. Defaults to None.
        masks (list[str], optional): List of segmentation masks. Defaults to None.
        class_names (list[str], optional): List of class names. Defaults to None.
        classes_to_idx (list[str], optional): Mapping from class names to class indices. Defaults
            to None.
        transform (Callable | list[Callable], optional): A function/transform that takes in a
            PIL image and returns a transformed version. If a list of transforms is provided, they
            are applied depending on the target label. Defaults to None.
        target_transform (Callable, optional): A function/transform that takes in the target and
            transforms it. Defaults to None.
        mask_preprocess (Callable, optional): A function/transform that takes in the mask and
            transforms it. Defaults to None.
        background_pixel_value (int, optional): Index of the background label. Defaults to None.
        unlabelled_pixel_value (int, optional): Index of the void label. Defaults to None.


     Attributes:
        class_names (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index, domain_index) tuples.
        images (list): List of paths to images.
        masks (list): List of segmentation masks.
    """

    def __init__(
        self,
        root: str,
        images: list[str] | None = None,
        masks: list[str] | None = None,
        class_names: list[str] | None = None,
        classes_to_idx: dict[str, int] | None = None,
        transform: Callable | list[Callable] | None = None,
        target_transform: Callable | None = None,
        mask_preprocess: Callable | None = None,
        background_pixel_value: int | None = None,
        unlabelled_pixel_value: int | None = None,
    ) -> None:
        self.samples = list(zip(images, masks))
        self.images = images
        self.masks = masks

        self.class_names = class_names
        self.classes_to_idx = classes_to_idx or {c: i for i, c in enumerate(self.class_names)}
        self.background_pixel_value = background_pixel_value
        self.unlabelled_pixel_value = unlabelled_pixel_value

        # map pixel values to class indices
        offset = 1 if background_pixel_value != 0 and unlabelled_pixel_value != 0 else 0
        self.pixel_to_classes = {i: i + offset for i in range(len(self.class_names))}
        if background_pixel_value and background_pixel_value != 0:
            self.pixel_to_classes[background_pixel_value] = 0
        if unlabelled_pixel_value and unlabelled_pixel_value != 0:
            self.pixel_to_classes[unlabelled_pixel_value] = 0

        if offset == 1:
            self.class_names = ["background"] + self.class_names

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mask_preprocess = mask_preprocess

        self.loader = default_loader

    def __getitem__(self, index: int) -> dict:
        path, mask_fp = self.samples[index]
        mask_pil = Image.open(mask_fp)

        mask_np = np.array(mask_pil)
        if self.mask_preprocess is not None:
            mask_np = self.mask_preprocess(mask_np)

        targets_idx = list(np.unique(mask_np))
        if self.background_pixel_value in targets_idx:
            targets_idx.remove(self.background_pixel_value)
        if self.unlabelled_pixel_value in targets_idx:
            targets_idx.remove(self.unlabelled_pixel_value)
        if len(targets_idx) > 0:
            targets_idx = list(np.vectorize(self.pixel_to_classes.get)(targets_idx))
        targets_name = [self.class_names[t] for t in targets_idx]

        targets = torch.tensor(targets_idx, dtype=torch.long)
        image_pil = self.loader(path)
        if self.transform is not None:
            if isinstance(self.transform, list):
                image_tensor = self.transform[targets](image_pil)
            else:
                image_tensor = self.transform(image_pil)
        if self.target_transform is not None:
            targets = [self.target_transform(t) for t in targets]

        targets_one_hot = torch.zeros(len(self.class_names), dtype=torch.long)
        targets_one_hot[targets] = 1

        mask_np = np.vectorize(self.pixel_to_classes.get)(mask_np)

        data = {
            "images_fp": path,
            "images_pil": image_pil,
            "images_tensor": image_tensor,
            "targets_idx": targets_idx,
            "targets_one_hot": targets_one_hot,
            "targets_name": targets_name,
            "masks_fp": mask_fp,
            "masks_pil": mask_pil,
            "masks_np": mask_np,
        }

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        if self.class_names is not None:
            if len(self.class_names) > 10:
                body += [f"Classes: {', '.join(self.class_names[:10])}..."]
            else:
                body += [f"Classes: {', '.join(self.class_names)}"]
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


def default_loader(path: str) -> Any:
    """Loads an image from a path.

    Args:
        path (str): str to the image.

    Returns:
        PIL.Image: The image.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
