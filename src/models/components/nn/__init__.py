from src.models.components.nn.classifiers import NearestNeighboursClassifier
from src.models.components.nn.encoders import LanguageTransformer

__all__ = ["LanguageTransformer", "NearestNeighboursClassifier"]

CLASSIFIERS = {
    "nearest_neighbours": NearestNeighboursClassifier,
}

LANGUAGE_ENCODERS = {
    "transformer": LanguageTransformer,
}
