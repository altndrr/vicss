from src.models.cased import CaSED
from src.models.clip import CLIP
from src.models.dense_cased import DenseCaSED
from src.models.vocabulary_free_clip import VocabularyFreeCLIP

__all__ = ["CaSED", "CLIP", "DenseCaSED", "VocabularyFreeCLIP"]

MODELS = {
    "cased": CaSED,
    "clip": CLIP,
    "dense_cased": DenseCaSED,
    "vocabulary_free_clip": VocabularyFreeCLIP,
}
