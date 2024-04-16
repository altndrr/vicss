from pathlib import Path
from typing import Any

import pandas as pd

from src.data._base import BaseDataModule
from src.data._utils import download_data, extract_data
from src.data.components.datasets import SegmentationDataset


class PascalVOC20(BaseDataModule):
    """LightningDataModule for PascalVOC-20 dataset.

    Statistics:
        - 2,913 images.
        - 20 classes (+1 background class).
        - URL: http://host.robots.ox.ac.uk/pascal/VOC/.

    Reference:
        - Everingham et al. The Pascal Visual Object Classes (VOC) Challenge. IJCV 2010.

    Args:
        data_dir (str): Path to the data directory. Defaults to "data/".
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    name: str = "PascalVOC20"
    task: str = "segmentation"

    classes: list[str] = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv monitor",
    ]

    alt_name: str = "pascal_voc_20"
    data_url: dict = {
        "images": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "seg_class": "http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip",
        "seg_class_vis": "http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip",
        "list": "http://cs.jhu.edu/~cxliu/data/list.zip",
    }

    def __init__(
        self, *args, data_dir: str = "data/", artifact_dir: str = "artifacts/", **kwargs
    ) -> None:
        super().__init__(*args, data_dir=data_dir, **kwargs)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes) or 21

    def prepare_data(self) -> None:
        """Download data if needed."""
        dataset_path = Path(self.hparams.data_dir, self.name)
        if dataset_path.exists():
            return

        for name, url in self.data_url.items():
            file_name = Path(url).name

            # download data
            target_path = Path(self.hparams.data_dir, file_name)
            download_data(url, target_path, from_gdrive=False)

            extract_data(target_path)
            dataset_path.mkdir(parents=True, exist_ok=True)
            if name == "images":
                output_path = Path(self.hparams.data_dir, "VOCdevkit/VOC2012")
                output_path.rename(dataset_path)
                Path(self.hparams.data_dir, "VOCdevkit").rmdir()
            else:
                folder_name = file_name.split(".")[0]
                output_path = Path(self.hparams.data_dir, folder_name)
                output_path.rename(dataset_path / folder_name)

    def setup(self, stage: str | None = None) -> None:
        """Load data.

        Set variables: `self.data_train` , `self.data_val` and `self.data_test`.
        """
        if self.data_train and self.data_val and self.data_test:
            return

        dataset_path = Path(self.hparams.data_dir, self.name)

        data = {}
        for split in ["train", "val", "test"]:
            text_split_filename = "train.txt" if split in ["train", "val"] else "val.txt"
            text_split_path = dataset_path / "list" / text_split_filename

            with open(text_split_path) as f:
                lines = f.readlines()

            lines = [line.replace("\n", "") for line in lines]
            lines = [line.split() for line in lines]

            file_paths = [str(dataset_path / line[0][1:]) for line in lines]
            masks_paths = [str(dataset_path / line[1][1:]) for line in lines]

            if split in ["train", "val"]:
                split_fp = Path(self.hparams.artifact_dir, "data", self.alt_name, "split.csv")
                split_df = pd.read_csv(split_fp)
                split_paths = split_df[split_df["split"] == split]["filename"]
                selected_filenames = split_paths.apply(lambda x: str(dataset_path / x)).tolist()
                selected_masks = split_paths.apply(lambda x: str(dataset_path / x))
                selected_masks = selected_masks.apply(
                    lambda x: x.replace("JPEGImages", "SegmentationClassAug").replace("jpg", "png")
                ).tolist()

                file_paths = [fp for fp in file_paths if fp in selected_filenames]
                masks_paths = [fp for fp in masks_paths if fp in selected_masks]

            data[split] = SegmentationDataset(
                str(dataset_path),
                images=file_paths,
                masks=masks_paths,
                class_names=self.classes,
                transform=self.preprocess,
                background_pixel_value=0,
                unlabelled_pixel_value=255,
            )

        self.data_train = data["train"]
        self.data_val = data["val"]
        self.data_test = data["test"]

    def teardown(self, stage: str | None = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = PascalVOC20()
