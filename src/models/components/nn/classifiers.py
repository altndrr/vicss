from functools import partial

import numpy as np
import torch


class NearestNeighboursClassifier(torch.nn.Module):
    """Nearest neighbours classifier.

    It computes the similarity between the query and the supports using the
    cosine similarity and then normalize to obtain the probabilities.

    Args:
        tau (float): Temperature for the softmax. Defaults to 1.0.
        out_norm (str): Normalization to apply to the logits. Defaults to "softmax".
    """

    def __init__(self, tau: float = 1.0, out_norm: str = "softmax") -> None:
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = torch.nn.Parameter(torch.zeros([]))
        self.tau = tau

        assert out_norm in ["softmax", "sigmoid"]
        self.out_norm = partial(torch.softmax, dim=-1) if out_norm == "softmax" else torch.sigmoid

    def forward(
        self, query: torch.Tensor, supports: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query (torch.Tensor): Query tensor.
            supports (torch.Tensor): Supports tensor.
            mask (torch.Tensor, optional): Zero out the similarities for the masked supports.
                Defaults to None.
        """
        query = query / query.norm(dim=-1, keepdim=True)
        supports = supports / supports.norm(dim=-1, keepdim=True)

        if supports.dim() == 2:
            supports = supports.unsqueeze(0)

        Q, _ = query.shape
        N, C, _ = supports.shape

        supports = supports.mean(dim=0)
        supports = supports / supports.norm(dim=-1, keepdim=True)
        similarity = self.logit_scale.exp() * query @ supports.T + self.logit_bias
        similarity = similarity / self.tau if self.tau != 1.0 else similarity

        if mask is not None:
            assert mask.shape[0] == query.shape[0] and mask.shape[1] == supports.shape[0]
            similarity = torch.masked_fill(similarity, mask == 0, float("-inf"))

        return self.out_norm(similarity)
