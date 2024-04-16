from pathlib import Path
from typing import Any

import pandas as pd

from src.data._base import BaseDataModule
from src.data._utils import download_data, extract_data
from src.data.components.datasets import SegmentationDataset


class ADE20K150(BaseDataModule):
    """LightningDataModule for ADE20K-150 dataset.

    Statistics:
        - 22,210 images.
        - 150 classes (+1 background class).
        - URL: https://groups.csail.mit.edu/vision/datasets/ADE20K/.

    Reference:
        - Zhou et al. Semantic Understanding of Scenes through ADE20K Dataset. IJCV 2018.
        - Zhou et al. Scene Parsing through ADE20K Dataset. CVPR 2017.

    Args:
        data_dir (str): Path to the data directory. Defaults to "data/".
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    name: str = "ADE20K150"
    task: str = "segmentation"

    classes: list[str]

    alt_name: str = "ade20k_150"
    data_url: str = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

    def __init__(self, *args, data_dir: str = "data/", artifact_dir: str = "artifacts/", **kwargs):
        super().__init__(*args, data_dir=data_dir, **kwargs)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes) or 151

    def prepare_data(self) -> None:
        """Download data if needed."""
        dataset_path = Path(self.hparams.data_dir, self.name)
        if dataset_path.exists():
            return

        # download data
        target_path = Path(self.hparams.data_dir, "ade20k.zip")
        download_data(self.data_url, target_path, from_gdrive=False)

        # rename parent folder
        extract_data(target_path)
        output_path = Path(self.hparams.data_dir, "ADEChallengeData2016")
        output_path.rename(dataset_path)

    def setup(self, stage: str | None = None) -> None:
        """Load data.

        Set variables: `self.data_train` , `self.data_val` and `self.data_test`.
        """
        if self.data_train and self.data_val and self.data_test:
            return

        dataset_path = Path(self.hparams.data_dir, self.name)
        metadata_fp = Path(self.hparams.artifact_dir, "data", self.alt_name, "metadata.csv")

        metadata_df = pd.read_csv(metadata_fp)
        class_names = metadata_df["class_name"].to_list()

        data = {}
        for split in ["train", "val", "test"]:
            split_subdir = "training" if split in ["train", "val"] else "validation"
            masks = sorted(list(dataset_path.glob(f"annotations/{split_subdir}/*.png")))
            filenames = sorted(list(dataset_path.glob(f"images/{split_subdir}/*.jpg")))

            masks = [str(mask) for mask in masks]
            filenames = [str(filename) for filename in filenames]

            if split in ["train", "val"]:
                split_fp = Path(self.hparams.artifact_dir, "data", self.alt_name, "split.csv")
                split_df = pd.read_csv(split_fp)
                split_paths = split_df[split_df["split"] == split]["filename"]
                selected_filenames = split_paths.apply(lambda x: str(dataset_path / x)).tolist()
                selected_masks = split_paths.apply(lambda x: str(dataset_path / x))
                selected_masks = selected_masks.apply(
                    lambda x: x.replace("images", "annotations").replace(".jpg", ".png")
                ).tolist()

                filenames = [fp for fp in filenames if fp in selected_filenames]
                masks = [fp for fp in masks if fp in selected_masks]

            data[split] = SegmentationDataset(
                str(dataset_path),
                images=filenames,
                masks=masks,
                class_names=class_names,
                transform=self.preprocess,
                unlabelled_pixel_value=0,
            )

        # store the class names
        self.classes = class_names

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
    _ = ADE20K150()
