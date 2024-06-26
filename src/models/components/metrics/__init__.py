from src.models.components.metrics.aggregation import UniqueValues
from src.models.components.metrics.classification import SemanticClusterAccuracy
from src.models.components.metrics.segmentation import SemanticJaccardIndex, SemanticRecall
from src.models.components.metrics.text import SentenceIOU, SentenceScore

__all__ = [
    "SemanticClusterAccuracy",
    "SemanticJaccardIndex",
    "SemanticRecall",
    "SentenceScore",
    "SentenceIOU",
    "UniqueValues",
]

AGGREGATION = {
    "unique_values": UniqueValues,
}

CLASSIFICATION = {
    "semantic_cluster_accuracy": SemanticClusterAccuracy,
}

SEGMENTATION = {
    "semantic_jaccard_index": SemanticJaccardIndex,
    "semantic_recall": SemanticRecall,
}

TEXT = {
    "sentence_iou": SentenceIOU,
    "sentence_score": SentenceScore,
}
