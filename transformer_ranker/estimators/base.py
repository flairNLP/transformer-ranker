from abc import ABC, abstractmethod
from typing import Optional

import torch


class Estimator(ABC):
    """Abstract base class for transferability metrics."""

    def __init__(self, regression: bool, **kwargs):
        self.regression: bool = regression
        self.score: Optional[float] = None

    @abstractmethod
    def fit(self, *, embeddings: torch.Tensor, labels: torch.Tensor, **kwargs) -> float:
        """Compute score given embeddings and labels.

        :param embeddings: Embedding tensor (num_samples, num_features)
        :param labels: Label tensor (num_samples,)
        """
        pass
