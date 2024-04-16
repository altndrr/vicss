import json
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    LlavaForConditionalGeneration,
)

from src.models.components.retrieval import RetrievalDatabase, download_retrieval_databases

__all__ = ["BLIP2VQAVocabulary", "ImageNetVocabulary", "LLaVaVocabulary", "RetrievalVocabulary"]


class BaseVocabulary(ABC, torch.nn.Module):
    """Base class for a vocabulary for image classification."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> list[list[str]] | list[torch.Tensor]:
        values = self.forward(*args, **kwargs)
        return values

    @abstractmethod
    def forward(self, *args, **kwargs) -> list[list[str]] | list[torch.Tensor]:
        """Forward pass."""
        raise NotImplementedError


class BLIP2VQAVocabulary(BaseVocabulary):
    """Vocabulary based on VQA with BLIP2 on images.

    Args:
        model_name (str): Name of the model to use.
        question (str): Question to ask the model. Defaults to "Question: what's in the image?
            Answer:".
    """

    def __init__(
        self,
        *args,
        model_name: str = "blip2-flan-t5-xl",
        question: str = "Question: what's in the image? Answer:",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.question = question

        self._model = Blip2ForConditionalGeneration.from_pretrained(
            f"Salesforce/{model_name}", torch_dtype=torch.float16
        )

        processor = Blip2Processor.from_pretrained(f"Salesforce/{model_name}")
        self._preprocess = partial(processor, text=self.question, return_tensors="pt")
        self._decode = partial(processor.decode, skip_special_tokens=True)

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return list(self._model.parameters())[0].device

    @torch.no_grad()
    def forward(
        self,
        *args,
        images_pil: list[PIL.Image.Image] | None = None,
        images_fp: list[str] | None = None,
        **kwargs,
    ) -> list[list[str]] | list[torch.Tensor]:
        """Create a vocabulary for a batch of images.

        Args:
            images_pil (list[PIL.Image.Image]): Images to create vocabularies for. Defaults to
                None.
            images_fp (list[str]): Path to image files to create vocabularies for. Defaults to
                None.
        """
        assert images_pil is not None or images_fp is not None

        images = images_pil
        if images is None:
            images = [Image.open(fp).convert("RGB") for fp in images_fp]

        # generate captions
        inputs = self._preprocess(images).to(self.device)
        inputs["input_ids"] = inputs["input_ids"].repeat(len(images), 1)
        inputs["attention_mask"] = inputs["attention_mask"].repeat(len(images), 1)
        outputs = self._model.generate(**inputs, max_new_tokens=30)

        # split captions into vocabularies
        vocabularies = [list(set(self._decode(output).split(","))) for output in outputs]

        # clean vocabularies
        vocabularies = [[v.strip().lower() for v in vocabulary] for vocabulary in vocabularies]
        vocabularies = [[v for v in vocabulary if v] for vocabulary in vocabularies]

        # fill empty vocabularies with a single word
        vocabularies = [["object"] if not v else v for v in vocabularies]

        return vocabularies


class ImageNetVocabulary(BaseVocabulary):
    """Vocabulary based on ImageNet classes.

    Args:
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    def __init__(self, *args, artifact_dir: str = "artifacts/", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._artifact_dir = artifact_dir

        metadata_fp = str(Path(self._artifact_dir, "data", "imagenet", "metadata.csv"))
        metadata_df = pd.read_csv(metadata_fp)
        class_names = metadata_df["class_name"].tolist()

        self._words = class_names

    def forward(self, *args, **kwargs) -> list[list[str]]:
        """Create a vocabulary for a batch of images."""
        batch_size = max(len(kwargs.get("images_z", kwargs.get("images_fp", []))), 1)

        return [self._words] * batch_size


class LLaVaVocabulary(BaseVocabulary):
    """Vocabulary based on LLaVa.

    Args:
        model_name (str): Name of the model to use.
        question (str): Question to ask the model. The question is formatted with the template
            "<image>\nUSER: {question}\nASSISTANT:". Defaults to "What's the content of the
            image?".
    """

    def __init__(
        self,
        *args,
        model_name: str = "llava-1.5-7b-hf",
        question: str = "What's the content of the image?",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._template = "<image>\nUSER: {question}\nASSISTANT:"
        self.model_name = model_name
        self.question = question

        self._model = LlavaForConditionalGeneration.from_pretrained(
            f"llava-hf/{model_name}", torch_dtype=torch.float16
        )

        processor = AutoProcessor.from_pretrained(f"llava-hf/{model_name}")
        processor.tokenizer.padding_side = "left"  # improve batch generation
        text = self._template.format(question=self.question)
        self._preprocess = partial(processor, text=text, return_tensors="pt")
        self._decode = partial(
            processor.batch_decode, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return list(self._model.parameters())[0].device

    @torch.no_grad()
    def forward(
        self,
        *args,
        images_pil: list[PIL.Image.Image] | None = None,
        images_fp: list[str] | None = None,
        **kwargs,
    ) -> list[list[str]] | list[torch.Tensor]:
        """Create a vocabulary for a batch of images.

        Args:
            images_pil (list[PIL.Image.Image]): Images to create vocabularies for. Defaults to
                None.
            images_fp (list[str]): Path to image files to create vocabularies for. Defaults to
                None.
        """
        assert images_pil is not None or images_fp is not None

        images = images_pil
        if images is None:
            images = [Image.open(fp).convert("RGB") for fp in images_fp]

        # generate captions
        inputs = self._preprocess(images=images).to(self.device)
        inputs["input_ids"] = inputs["input_ids"].repeat(len(images), 1)
        inputs["attention_mask"] = inputs["attention_mask"].repeat(len(images), 1)
        outputs = self._model.generate(**inputs, max_new_tokens=30)
        captions = self._decode(outputs)
        captions = [cap.split("ASSISTANT: ")[1] for cap in captions]  # keep only from "ASSISTANT:"

        # split captions into vocabularies
        vocabularies = [list(set(cap.split(","))) for cap in captions]

        # clean vocabularies
        vocabularies = [[v.strip().lower() for v in vocabulary] for vocabulary in vocabularies]
        vocabularies = [[v for v in vocabulary if v] for vocabulary in vocabularies]

        # fill empty vocabularies with a single word
        vocabularies = [["object"] if not v else v for v in vocabularies]

        return vocabularies


class RetrievalVocabulary(BaseVocabulary):
    """Vocabulary based on captions from an external database.

    Args:
        database_name (str): Name of the database to use.
        databases_dict_fp (str): Path to the databases dictionary file.
        num_samples (int): Number of samples to return. Default is 40.
    """

    def __init__(
        self, *args, database_name: str, databases_dict_fp: str, num_samples: int = 10, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.database_name = database_name
        self.databases_dict_fp = databases_dict_fp
        self.num_samples = num_samples

        with open(databases_dict_fp, encoding="utf-8") as f:
            databases_dict = json.load(f)

        download_retrieval_databases()
        self.database = RetrievalDatabase(databases_dict[database_name])

    def __call__(self, *args, **kwargs) -> list[list[str]]:
        values = super().__call__(*args, **kwargs)

        # keep only the `num_samples` first words
        num_samples = self.num_samples
        values = [value[:num_samples] for value in values]

        return values

    def forward(
        self, *args, images_z: torch.Tensor | None = None, **kwargs
    ) -> list[list[str]] | list[torch.Tensor]:
        """Create a vocabulary for a batch of images.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
        """
        assert images_z is not None

        images_z = images_z / images_z.norm(dim=-1, keepdim=True)
        images_z = images_z.cpu().detach().numpy().tolist()

        if isinstance(images_z[0], float):
            images_z = [images_z]

        query = np.matrix(images_z).astype("float32")
        results = self.database.query(query, modality="text", num_samples=self.num_samples)
        vocabularies = [[r["caption"] for r in result] for result in results]

        return vocabularies
