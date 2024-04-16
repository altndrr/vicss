from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassRecall

from src import utils


class SemanticJaccardIndex:
    """Semantic jaccard index metric factory.

    Args:
        mode (str): Mode to use. Either "hard", "soft", or "overlap", "nearest".
    """

    def __new__(cls, mode: str, *args, **kwargs) -> Metric:
        if mode == "hard":
            return SemanticHardJaccardIndex(*args, **kwargs)
        elif mode == "soft":
            return SemanticSoftJaccardIndex(*args, **kwargs)
        elif mode == "overlap":
            return SemanticClusterJaccardIndex(*args, match="overlap", **kwargs)
        elif mode == "nearest":
            return SemanticClusterJaccardIndex(*args, match="nearest", **kwargs)
        raise ValueError(f"Invalid mode {mode}")


class SemanticRecall:
    """Semantic recall metric factory.

    Args:
        mode (str): Mode to use. Either "hard", "soft", or "overlap", "nearest".
    """

    def __new__(cls, mode: str, *args, **kwargs) -> Metric:
        if mode == "hard":
            return SemanticHardRecall(*args, **kwargs)
        elif mode == "soft":
            return SemanticSoftRecall(*args, **kwargs)
        raise ValueError(f"Invalid mode {mode}")


class SemanticClusterJaccardIndex(Metric):
    """Metric to evaluate the cluster Jaccard index.

    It takes as input semantic masks composed of a list of class names and a corresponding
    semantic mask. Since predictions and class names may differ, the metric first matches
    each predicted class with the target class with the maximum co-occurrences. Then, it
    computes the intersection and union between the predicted semantic masks and the target
    semantic masks.

    Args:
        classes (list[str]): List of class names.
        average (str): Type of averaging to perform. Can be "micro" or "macro". Defaults to
            "micro".
        match (str): Mode to use to assign each predicted class to the target class.
            Can be "overlap" or "nearest". Defaults to "overlap".
    """

    def __init__(
        self, *args, classes: list[str], average: str = "micro", match: str = "overlap", **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert average in ["micro", "macro"]
        assert match in ["overlap", "nearest"]
        self.average = average
        self.match = match
        self.classes = classes
        self.encoder = utils.SentenceBERT()
        classes_z = self.encoder(classes)
        self.encoder.register_buffer("classes_z", classes_z, exists_ok=True)

        self.add_state("intersection", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("target_idx", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, values: list[tuple[list, np.ndarray]], targets: list[np.ndarray]) -> None:
        """Update state with data.

        Args:
            values (list[tuple[list, np.ndarray]]): Predicted semantic masks. The first element
                of the tuple is a list of class names, while the second element is the predicted
                semantic mask.
            targets (list[np.ndarray]): Targets masks.
        """
        intersections, unions = [], []
        target_idxs = []

        # if necessary, compute the embeddings of the class names
        if self.match == "nearest":
            classes_z = self.encoder.classes_z.unsqueeze(0).to(self.device)

            values_names = sum([names for names, _ in values], [])
            values_names_per_value = [len(names) for names, _ in values]
            values_names_z = self.encoder(values_names)
            if values_names_z.dim() == 1:
                values_names_z = values_names_z.unsqueeze(0)
            values_names_z = values_names_z.unsqueeze(1)
            values_names_z = torch.split(values_names_z, values_names_per_value)

        for i, (value, target) in enumerate(zip(values, targets)):
            target = torch.tensor(target, device=self.device)
            _, value_mask = value
            value_mask = torch.tensor(value_mask, dtype=torch.float, device=self.device)
            value_mask = F.interpolate(
                value_mask.unsqueeze(0), size=target.shape, mode="bilinear", align_corners=False
            )
            value_mask = value_mask.argmax(dim=1).squeeze(0)

            # assign predicted classes to target classes and compute scores
            if self.match == "overlap":
                matrix_size = (value_mask.max() + 1, len(self.classes))
                v, t = value_mask.view(-1), target.view(-1)
                co_occurrences = torch.zeros(matrix_size, dtype=torch.long, device=self.device)
                co_occurrences = torch.bincount(
                    v * matrix_size[1] + t, minlength=matrix_size[0] * matrix_size[1]
                ).view(matrix_size)
                value_mask = co_occurrences.argmax(dim=-1)[value_mask]
            elif self.match == "nearest":
                value_names_z = values_names_z[i]
                similarity = F.relu(F.cosine_similarity(classes_z, value_names_z, dim=-1))
                similarity[:, ~torch.unique(target)] = 0
                value_mask = similarity.argmax(dim=-1)[value_mask]

            # compute intersection and union
            matches = (value_mask == target).long()
            for idx in torch.unique(target):
                if idx == 0:
                    continue

                intersection = torch.sum(matches[target == idx])
                union = torch.sum(value_mask == idx) + torch.sum(target == idx) - intersection

                intersections.append(intersection)
                unions.append(union)
                target_idxs.append(idx)

        intersections = torch.tensor(intersections, device=self.device)
        unions = torch.tensor(unions, device=self.device)
        target_idxs = torch.tensor(target_idxs, device=self.device)

        self.intersection = torch.cat([self.intersection, intersections])
        self.union = torch.cat([self.union, unions])
        self.target_idx = torch.cat([self.target_idx, target_idxs])

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        if self.average == "micro":
            return torch.mean(self.intersection.float() / self.union.float())
        elif self.average == "macro":
            jaccard_indexes = []
            for idx in torch.unique(self.target_idx):
                mask = self.target_idx == idx
                class_intersection = self.intersection[mask].float()
                class_union = self.union[mask].float()
                jaccard_indexes.append(torch.mean(class_intersection / class_union))

            return torch.mean(torch.stack(jaccard_indexes))


class SemanticHardJaccardIndex(MulticlassJaccardIndex):
    """Metric to evaluate the semantic Jaccard index.

    Extends the original `torchmetrics.classification.MulticlassJaccardIndex` to support
    semantic masks composed of a list of class names and a corresponding semantic mask.

    Args:
        classes (list[str]): List of class names.
    """

    def __init__(self, *args, classes: list[str], **kwargs) -> None:
        super().__init__(*args, num_classes=len(classes), **kwargs)
        self.classes = classes
        self.class_to_idx = defaultdict(lambda: 0)
        self.class_to_idx.update({c: i for i, c in enumerate(classes)})

    def update(self, values: list[tuple[list, np.ndarray]], targets: list[np.ndarray]) -> None:
        """Update state with data.

        Args:
            values (list[tuple[list, np.ndarray]]): Predicted semantic masks. The first element
                of the tuple is a list of class names, while the second element is the predicted
                semantic mask.
            targets (list[np.ndarray]): Targets masks.
        """
        for value, target in zip(values, targets):
            names, value_mask = value

            value_mask = torch.tensor(value_mask, dtype=torch.float).unsqueeze(0)
            value_mask = F.interpolate(
                value_mask, size=target.shape, mode="bilinear", align_corners=False
            )
            value_mask = value_mask.argmax(dim=1).squeeze(0).numpy()

            values_idx_to_class_idx = defaultdict(lambda: len(self.classes))
            values_idx_to_class_idx.update({i: self.class_to_idx[v] for i, v in enumerate(names)})
            value_mask = np.vectorize(values_idx_to_class_idx.get)(value_mask)

            value = torch.tensor(value_mask, device=self.device)
            target = torch.tensor(target, device=self.device)

            super().update(value, target)


class SemanticSoftJaccardIndex(Metric):
    """Metric to evaluate the semantic soft Jaccard index.

    It takes as input semantic masks composed of a list of class names and a corresponding
    semantic mask. Since predictions and class names may differ, the metric computes the
    intersection and union between the predicted semantic masks and the target semantic masks
    by considering scores instead of binary values. The scores are computed as the cosine
    similarity between the predicted semantic mask and the semantic mask of each class.

    Args:
        classes (list[str]): List of class names.
        average (str): Type of averaging to perform. Can be "micro" or "macro". Defaults to
            "micro".
    """

    def __init__(self, *args, classes: list[str], average: str = "micro", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert average in ["micro", "macro"]
        self.classes = classes
        self.average = average
        self.encoder = utils.SentenceBERT()
        classes_z = self.encoder(classes)
        self.encoder.register_buffer("classes_z", classes_z, exists_ok=True)

        self.add_state("intersection", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("target_idx", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, values: list[tuple[list, np.ndarray]], targets: list[np.ndarray]) -> None:
        """Update state with data.

        Args:
            values (list[tuple[list, np.ndarray]]): Predicted semantic masks. The first element
                of the tuple is a list of class names, while the second element is the predicted
                semantic mask.
            targets (list[np.ndarray]): Targets masks.
        """
        intersections, unions = [], []
        target_idxs = []

        classes_z = self.encoder.classes_z.unsqueeze(0).to(self.device)

        values_names = sum([names for names, _ in values], [])
        values_names_per_value = [len(names) for names, _ in values]
        values_names_z = self.encoder(values_names)
        if values_names_z.dim() == 1:
            values_names_z = values_names_z.unsqueeze(0)
        values_names_z = values_names_z.unsqueeze(1)
        values_names_z = torch.split(values_names_z, values_names_per_value)

        for value, value_names_z, target in zip(values, values_names_z, targets):
            target = torch.tensor(target, device=self.device)
            _, value_mask = value
            value_mask = torch.tensor(value_mask, dtype=torch.float, device=self.device)
            value_mask = F.interpolate(
                value_mask.unsqueeze(0), size=target.shape, mode="bilinear", align_corners=False
            )
            value_mask = value_mask.argmax(dim=1).squeeze(0)

            similarity = F.relu(F.cosine_similarity(classes_z, value_names_z, dim=-1))
            value_scores = similarity[:, torch.unique(target)][value_mask].permute(2, 0, 1)

            for i, idx in enumerate(torch.unique(target)):
                if idx == 0:
                    continue

                mask = target == idx

                intersection = torch.sum(value_scores[i][mask])
                union = torch.sum(value_scores[i]) + torch.sum(mask) - intersection

                intersections.append(intersection)
                unions.append(union)
                target_idxs.append(idx)

        intersections = torch.tensor(intersections, device=self.device)
        unions = torch.tensor(unions, device=self.device)
        target_idxs = torch.tensor(target_idxs, device=self.device)

        self.intersection = torch.cat([self.intersection, intersections])
        self.union = torch.cat([self.union, unions])
        self.target_idx = torch.cat([self.target_idx, target_idxs])

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        if self.average == "micro":
            return torch.mean(self.intersection.float() / self.union.float())
        elif self.average == "macro":
            jaccard_indexes = []
            for idx in torch.unique(self.target_idx):
                mask = self.target_idx == idx
                class_intersection = self.intersection[mask].float()
                class_union = self.union[mask].float()
                jaccard_indexes.append(torch.mean(class_intersection / class_union))

            return torch.mean(torch.stack(jaccard_indexes))


class SemanticHardRecall(MulticlassRecall):
    """Metric to evaluate the semantic recall score.

    Extends the original `torchmetrics.classification.MulticlassRecall` to support semantic masks
    composed of a list of class names and a corresponding semantic mask.

    Args:
        classes (list[str]): List of class names.
    """

    def __init__(self, *args, classes: list[str], **kwargs) -> None:
        super().__init__(*args, num_classes=len(classes), **kwargs)
        self.classes = classes
        self.class_to_idx = defaultdict(lambda: 0)
        self.class_to_idx.update({c: i for i, c in enumerate(classes)})

    def update(self, values: list[tuple[list, np.ndarray]], targets: list[np.ndarray]) -> None:
        """Update state with data.

        Args:
            values (list[tuple[list, np.ndarray]]): Predicted semantic masks. The first element
                of the tuple is a list of class names, while the second element is the predicted
                semantic mask.
            targets (list[np.ndarray]): Targets masks.
        """
        for value, target in zip(values, targets):
            names, value_mask = value

            value_mask = torch.tensor(value_mask, dtype=torch.float).unsqueeze(0)
            value_mask = F.interpolate(
                value_mask, size=target.shape, mode="bilinear", align_corners=False
            )
            value_mask = value_mask.argmax(dim=1).squeeze(0).numpy()

            values_idx_to_class_idx = defaultdict(lambda: len(self.classes))
            values_idx_to_class_idx.update({i: self.class_to_idx[v] for i, v in enumerate(names)})
            value_mask = np.vectorize(values_idx_to_class_idx.get)(value_mask)

            value = torch.tensor(value_mask, device=self.device)
            target = torch.tensor(target, device=self.device)

            super().update(value, target)


class SemanticSoftRecall(Metric):
    """Metric to evaluate the semantic soft recall score.

    It takes as input semantic masks composed of a list of class names and a corresponding
    semantic mask. Since predictions and class names may differ, the metric computes the
    intersection and union between the predicted semantic masks and the target semantic masks
    by considering scores instead of binary values. The scores are computed as the cosine
    similarity between the predicted semantic mask and the semantic mask of each class.

    Args:
        classes (list[str]): List of class names.
        average (str): Type of averaging to perform. Can be "micro" or "macro". Defaults to
            "micro".
    """

    def __init__(self, *args, classes: list[str], average: str = "micro", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert average in ["micro", "macro"]
        self.classes = classes
        self.average = average
        self.encoder = utils.SentenceBERT()
        classes_z = self.encoder(classes)
        self.encoder.register_buffer("classes_z", classes_z, exists_ok=True)

        self.add_state("recall", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("target_idx", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, values: list[tuple[list, np.ndarray]], targets: list[np.ndarray]) -> None:
        """Update state with data.

        Args:
            values (list[tuple[list, np.ndarray]]): Predicted semantic masks. The first element
                of the tuple is a list of class names, while the second element is the predicted
                semantic mask.
            targets (list[np.ndarray]): Targets masks.
        """
        classes_z = self.encoder.classes_z.unsqueeze(0).to(self.device)

        values_names = sum([names for names, _ in values], [])
        values_names_per_value = [len(names) for names, _ in values]
        values_names_z = self.encoder(values_names)
        if values_names_z.dim() == 1:
            values_names_z = values_names_z.unsqueeze(0)
        values_names_z = values_names_z.unsqueeze(1)
        values_names_z = torch.split(values_names_z, values_names_per_value)

        recall = []
        target_idxs = []
        for value, value_names_z, target in zip(values, values_names_z, targets):
            target = torch.tensor(target, device=self.device)
            _, value_mask = value
            value_mask = torch.tensor(value_mask, dtype=torch.float, device=self.device)
            value_mask = F.interpolate(
                value_mask.unsqueeze(0), size=target.shape, mode="bilinear", align_corners=False
            )
            value_mask = value_mask.argmax(dim=1).squeeze(0)

            similarity = F.relu(F.cosine_similarity(classes_z, value_names_z, dim=-1))

            rows, cols = value_mask.flatten(), target.flatten()
            value_scores = similarity[rows, cols].reshape(value_mask.shape)

            for idx in torch.unique(target):
                if idx == 0:
                    continue

                mask = target == idx

                recall.append(torch.sum(value_scores[mask]) / torch.sum(mask))
                target_idxs.append(idx)

        recall = torch.tensor(recall, device=self.device)
        target_idxs = torch.tensor(target_idxs, device=self.device)

        self.recall = torch.cat([self.recall, recall])
        self.target_idx = torch.cat([self.target_idx, target_idxs])

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        if self.average == "micro":
            return torch.mean(self.recall)
        elif self.average == "macro":
            recalls = []
            for idx in torch.unique(self.target_idx):
                mask = self.target_idx == idx
                recalls.append(torch.mean(self.recall[mask]))

            return torch.mean(torch.stack(recalls))
