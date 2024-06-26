import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import inflect
import nltk
import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from flair.data import Sentence
from flair.models import SequenceTagger

__all__ = [
    "DynamicResize",
    "DropFileExtensions",
    "DropNonAlpha",
    "DropShortWords",
    "DropSpecialCharacters",
    "DropTokens",
    "DropURLs",
    "DropWords",
    "FilterPOS",
    "FrequencyMinWordCount",
    "FrequencyTopK",
    "ReplaceSeparators",
    "ToMultiCropTensors",
    "ToRGBTensor",
    "ToLowercase",
    "ToSingular",
]


class BaseTextTransform(ABC):
    """Base class for string transforms."""

    @abstractmethod
    def __call__(self, text: str) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DynamicResize(T.Resize):
    """Resize the input PIL Image to the given size.

    Extends the torchvision Resize transform to dynamically evaluate the second dimension of the
    output size based on the aspect ratio of the first input image.
    """

    def forward(self, img) -> PIL.Image.Image:
        """Forward pass of the transform.

        Args:
            img (PIL Image): Image to resize.
        """
        if isinstance(self.size, int):
            _, h, w = F.get_dimensions(img)
            aspect_ratio = w / h
            side = self.size

            if aspect_ratio < 1.0:
                self.size = int(side / aspect_ratio), side
            else:
                self.size = side, int(side * aspect_ratio)

        return super().forward(img)


class DropFileExtensions(BaseTextTransform):
    """Remove file extensions from the input text."""

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove file extensions from.
        """
        text = re.sub(r"\.\w+", "", text)

        return text


class DropNonAlpha(BaseTextTransform):
    """Remove non-alpha words from the input text."""

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove non-alpha words from.
        """
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        return text


class DropShortWords(BaseTextTransform):
    """Remove short words from the input text.

    Args:
        min_length (int): Minimum length of words to keep.
    """

    def __init__(self, min_length) -> None:
        super().__init__()
        self.min_length = min_length

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove short words from.
        """
        text = " ".join([word for word in text.split() if len(word) >= self.min_length])

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_length={self.min_length})"


class DropSpecialCharacters(BaseTextTransform):
    """Remove special characters from the input text.

    Special characters are defined as any character that is not a word character, whitespace,
    hyphen, period, apostrophe, or ampersand.
    """

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove special characters from.
        """
        text = re.sub(r"[^\w\s\-\.\'\&]", "", text)

        return text


class DropTokens(BaseTextTransform):
    """Remove tokens from the input text.

    Tokens are defined as strings enclosed in angle brackets, e.g. <token>.
    """

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove tokens from.
        """
        text = re.sub(r"<[^>]+>", "", text)

        return text


class DropURLs(BaseTextTransform):
    """Remove URLs from the input text."""

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove URLs from.
        """
        text = re.sub(r"http\S+", "", text)

        return text


class DropWords(BaseTextTransform):
    """Remove words from the input text.

    It is case-insensitive and supports singular and plural forms of the words.
    """

    def __init__(self, words: list[str]) -> None:
        super().__init__()
        self.words = words
        self.pattern = r"\b(?:{})\b".format("|".join(words))

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove words from.
        """
        text = re.sub(self.pattern, "", text, flags=re.IGNORECASE)

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pattern={self.pattern})"


class FilterPOS(BaseTextTransform):
    """Filter words by POS tags.

    Args:
        tags (list): List of POS tags to remove.
        engine (str): POS tagger to use. Must be one of "nltk" or "flair". Defaults to "nltk".
        keep_compound_nouns (bool): Whether to keep composed words. Defaults to True.
    """

    def __init__(self, tags: list, engine: str = "nltk", keep_compound_nouns: bool = True) -> None:
        super().__init__()
        self.tags = tags
        self.engine = engine
        self.keep_compound_nouns = keep_compound_nouns

        if engine == "nltk":
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("punkt", quiet=True)
            self.tagger = lambda x: nltk.pos_tag(nltk.word_tokenize(x))
        elif engine == "flair":
            self.tagger = SequenceTagger.load("flair/pos-english-fast").predict

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove words with specific POS tags from.
        """
        if self.engine == "nltk":
            word_tags = self.tagger(text)
            text = " ".join([word for word, tag in word_tags if tag not in self.tags])
        elif self.engine == "flair":
            sentence = Sentence(text)
            self.tagger(sentence)
            text = " ".join([token.text for token in sentence.tokens if token.tag in self.tags])

        if self.keep_compound_nouns:
            compound_nouns = []

            if self.engine == "nltk":
                for i in range(len(word_tags) - 1):
                    if word_tags[i][1] == "NN" and word_tags[i + 1][1] == "NN":
                        # if they are the same word, skip
                        if word_tags[i][0] == word_tags[i + 1][0]:
                            continue

                        compound_noun = word_tags[i][0] + "_" + word_tags[i + 1][0]
                        compound_nouns.append(compound_noun)
            elif self.engine == "flair":
                for i in range(len(sentence.tokens) - 1):
                    if sentence.tokens[i].tag == "NN" and sentence.tokens[i + 1].tag == "NN":
                        # if they are the same word, skip
                        if sentence.tokens[i].text == sentence.tokens[i + 1].text:
                            continue

                        compound_noun = sentence.tokens[i].text + "_" + sentence.tokens[i + 1].text
                        compound_nouns.append(compound_noun)

            text = " ".join([text, " ".join(compound_nouns)])

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tags={self.tags}, engine={self.engine})"


class FrequencyMinWordCount(BaseTextTransform):
    """Keep only words that occur more than a minimum number of times in the input text.

    If the threshold is too strong and no words pass the threshold, the threshold is reduced to
    the most frequent word.

    Args:
        min_count (int): Minimum number of occurrences of a word to keep.
    """

    def __init__(self, min_count) -> None:
        super().__init__()
        self.min_count = min_count

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove infrequent words from.
        """
        if self.min_count <= 1:
            return text

        words = text.split()
        word_counts = {word: words.count(word) for word in words}

        # if nothing passes the threshold, reduce the threshold to the most frequent word
        max_word_count = max(word_counts.values() or [0])
        min_count = max_word_count if self.min_count > max_word_count else self.min_count

        text = " ".join([word for word in words if word_counts[word] >= min_count])

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_count={self.min_count})"


class FrequencyTopK(BaseTextTransform):
    """Keep only the top k most frequent words in the input text.

    In case of a tie, all words with the same count as the last word are kept.

    Args:
        top_k (int): Number of top words to keep.
    """

    def __init__(self, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove infrequent words from.
        """
        if self.top_k < 1:
            return text

        words = text.split()
        word_counts = {word: words.count(word) for word in words}
        top_words = sorted(word_counts, key=word_counts.get, reverse=True)

        # in case of a tie, keep all words with the same count
        top_words = top_words[: self.top_k]
        top_words = [word for word in top_words if word_counts[word] == word_counts[top_words[-1]]]

        text = " ".join([word for word in words if word in top_words])

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"


class ReplaceSeparators(BaseTextTransform):
    """Replace underscores and dashes with spaces."""

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to replace separators in.
        """
        text = re.sub(r"[_\-]", " ", text)

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RemoveDuplicates(BaseTextTransform):
    """Remove duplicate words from the input text."""

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to remove duplicate words from.
        """
        text = " ".join(list(set(text.split())))

        return text


class TextCompose:
    """Compose several transforms together.

    It differs from the torchvision.transforms.Compose class in that it applies the transforms to
    a string instead of a PIL Image or Tensor. In addition, it automatically join the list of
    input strings into a single string and splits the output string into a list of words.

    Args:
        transforms (list): List of transforms to compose.
    """

    def __init__(self, transforms: list[BaseTextTransform]) -> None:
        self.transforms = transforms

    def __call__(self, text: str | list[str]) -> list[str]:
        """
        Args:
            text (str | list[str]): Text to transform.
        """
        if isinstance(text, list):
            text = " ".join(text)

        for t in self.transforms:
            text = t(text)
        return text.split()

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ToRGBTensor(T.ToTensor):
    """Convert a `PIL Image` or `numpy.ndarray` to tensor.

    Compared with the torchvision `ToTensor` transform, it converts images with a single channel to
    RGB images. In addition, the conversion to tensor is done only if the input is not already a
    tensor.
    """

    def __call__(self, pic: PIL.Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Args:
            pic (PIL Image | numpy.ndarray | torch.Tensor): Image to be converted to tensor.
        """
        img = pic if isinstance(pic, torch.Tensor) else F.to_tensor(pic)
        img = cast(torch.Tensor, img)

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToMultiCropTensors:
    """Convert an image tensor to multiple cropped image tensors.

    It supports grid cropping with different grid sizes and random cropping. Multiple strategies
    can be combined together.

    Args:
        grid_sizes (int | list[int]): Grid sizes to use for grid cropping. Defaults to [2].
        num_random_crops (int): Number of random crops to extract. Defaults to 0.
        min_random_scale (float): Min size of the random crop as a fraction of the image size.
            Defaults to 0.08.
        max_random_scale (float): Max size of the random crop as a fraction of the image size.
            Defaults to 0.3.
        stride_grid (bool): Whether to also extract crops with a stride of 0.5. Defaults to False.
        output_size (int | tuple[int, int]): Size of returned patches. Defaults to (224, 224).
    """

    def __init__(
        self,
        grid_sizes: int | list[int] = [2],
        num_random_crops: int = 0,
        min_random_scale: float = 0.08,
        max_random_scale: float = 0.3,
        stride_grid: bool = False,
        output_size: int | tuple[int, int] = (224, 224),
    ) -> None:
        self.grid_sizes = [grid_sizes] if isinstance(grid_sizes, int) else grid_sizes
        self.num_random_crops = num_random_crops
        self.min_random_scale = min_random_scale
        self.max_random_scale = max_random_scale
        self.stride_grid = stride_grid
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Image tensor to convert to patch tensor.
        """
        all_crops = []
        all_bboxes = []

        for grid_size in self.grid_sizes:
            possible_strides = [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)]
            if not self.stride_grid or grid_size == 1:
                possible_strides = [(0.0, 0.0)]

            for stride_top, stride_left in possible_strides:
                grid_height, grid_width = grid_size, grid_size
                cell_height, cell_width = img.shape[1] // grid_height, img.shape[2] // grid_width
                img_size = (cell_height * grid_size, cell_width * grid_size)
                resized_img = F.resize(img, size=img_size, antialias=True)

                # store original dimensions tof bbox computation
                og_cell_height, og_cell_width = cell_height, cell_width
                og_img_size = img_size

                # crop the image by half the cell size to extract strided crops
                pad_height, pad_width = int(stride_top > 0.0), int(stride_left > 0.0)
                if pad_height or pad_width:
                    crop_top = cell_height // 2 if pad_height else 0
                    crop_left = cell_width // 2 if pad_width else 0
                    crop_height = resized_img.shape[1] - pad_height * cell_height
                    crop_width = resized_img.shape[2] - pad_width * cell_width
                    resized_img = F.crop(resized_img, crop_top, crop_left, crop_height, crop_width)

                    # overwrite cell sizes
                    grid_height = grid_size - int(stride_top > 0)
                    grid_width = grid_size - int(stride_left > 0)
                    cell_height = resized_img.shape[1] // grid_height
                    cell_width = resized_img.shape[2] // grid_width

                    # resize image to a multiple of the grid size
                    img_size = (cell_height * grid_height, cell_width * grid_width)
                    resized_img = F.resize(resized_img, size=img_size, antialias=True)

                # crop image and resize to output size
                crops = resized_img.unfold(1, cell_height, cell_height)
                crops = crops.unfold(2, cell_width, cell_width)
                crops = crops.reshape(3, -1, cell_height, cell_width).transpose(0, 1)
                crops = F.resize(crops, size=self.output_size, antialias=True)

                # compute bounding boxes
                offset_top = stride_top * og_cell_height
                offset_left = stride_left * og_cell_width
                bboxes = [
                    [
                        (offset_left + (j * og_cell_width)) / float(og_img_size[1]),
                        (offset_top + (i * og_cell_height)) / float(og_img_size[0]),
                        og_cell_width / float(og_img_size[1]),
                        og_cell_height / float(og_img_size[0]),
                    ]
                    for i in range(grid_height)
                    for j in range(grid_width)
                ]
                bboxes = torch.tensor(bboxes)

                all_crops.append(crops)
                all_bboxes.append(bboxes)

        random_crops, random_bboxes = [], []
        for _ in range(self.num_random_crops):
            # sample random crop size
            scale = np.random.uniform(self.min_random_scale, self.max_random_scale)
            _, h, w = F.get_dimensions(img)
            tw = int(img.shape[2] * scale)
            th = int(img.shape[1] * scale)

            if h < th or w < tw:
                raise ValueError(f"Crop size {(th, tw)} is larger than input image size {(h, w)}")

            if w == tw and h == th:
                return 0, 0, h, w

            i = torch.randint(0, h - th + 1, size=(1,)).item()
            j = torch.randint(0, w - tw + 1, size=(1,)).item()

            # crop image and resize to output size
            crop = F.resized_crop(img, i, j, th, tw, size=self.output_size, antialias=True)

            # compute bounding boxes
            bboxes = torch.tensor([j / w, i / h, tw / w, th / h])

            random_crops.append(crop)
            random_bboxes.append(bboxes)

        if len(random_crops) > 0:
            all_crops.append(torch.stack(random_crops))
            all_bboxes.append(torch.stack(random_bboxes))

        all_crops = torch.cat(all_crops, dim=0)
        all_bboxes = torch.cat(all_bboxes, dim=0)

        return {
            "images_tensor": F.resize(img, size=self.output_size, antialias=True),
            "images_crops_tensor": all_crops,
            "images_crops_bbox": all_bboxes,
        }


class ToLowercase(BaseTextTransform):
    """Convert text to lowercase."""

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to convert to lowercase.
        """
        text = text.lower()

        return text


class ToSingular(BaseTextTransform):
    """Convert plural words to singular form."""

    def __init__(self) -> None:
        super().__init__()
        self.transform = inflect.engine().singular_noun

    def __call__(self, text: str) -> str:
        """
        Args:
            text (str): Text to convert to singular form.
        """
        words = text.split()
        for i, word in enumerate(words):
            if not word.endswith("s"):
                continue

            if word[-2:] in ["ss", "us", "is"]:
                continue

            if word[-3:] in ["ies", "oes"]:
                continue

            words[i] = self.transform(word) or word

        text = " ".join(words)

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def default_image_preprocess(size: int | None = None) -> T.Compose:
    """Default preprocessing transforms for images.

    Args:
        size (int): Size to resize image to.
    """
    transforms = []
    if size is not None:
        transforms.append(DynamicResize(size, interpolation=T.InterpolationMode.BICUBIC))
    transforms.append(ToRGBTensor())
    transforms = T.Compose(transforms)

    return transforms


def default_text_preprocess() -> Callable:
    """Default preprocessing transforms for text."""

    def text_preprocess(texts: list[str], prompts: list[str] | None = None) -> list[list[str]]:
        prompts = prompts or ["{}"]
        texts = [text.replace("_", " ") for text in texts]
        texts_views = [[p.format(text) for text in texts] for p in prompts]
        return texts_views

    return text_preprocess


def default_vocab_transform() -> TextCompose:
    """Default transforms for vocabs."""
    words_to_drop = [
        "image",
        "photo",
        "picture",
        "thumbnail",
        "logo",
        "symbol",
        "clipart",
        "portrait",
        "painting",
        "illustration",
        "icon",
        "profile",
    ]
    pos_tags = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VBG", "VBN"]

    transforms = []
    transforms.append(DropTokens())
    transforms.append(DropURLs())
    transforms.append(DropSpecialCharacters())
    transforms.append(DropFileExtensions())
    transforms.append(ReplaceSeparators())
    transforms.append(DropShortWords(min_length=3))
    transforms.append(DropNonAlpha())
    transforms.append(ToLowercase())
    transforms.append(ToSingular())
    transforms.append(DropWords(words=words_to_drop))
    transforms.append(FrequencyMinWordCount(min_count=2))
    transforms.append(FilterPOS(tags=pos_tags, engine="flair", keep_compound_nouns=False))
    transforms.append(RemoveDuplicates())

    transforms = TextCompose(transforms)

    return transforms
