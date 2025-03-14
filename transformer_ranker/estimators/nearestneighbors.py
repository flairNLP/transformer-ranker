from typing import Union

import torch
from torch.nn.functional import cosine_similarity
from torchmetrics.functional import f1_score

from .base import Estimator


class NearestNeighbors(Estimator):
    def __init__(self, regression: bool = False, k: int = 3):
        """
        K-Nearest Neighbors estimator.

        :param k: Number of nearest neighbors to consider (3 by default).
        :param regression: Boolean flag if the task is regression.
        """
        super().__init__(regression=regression)

        self.k = k
        self.distance_metrics = {
            "euclidean": lambda x, y: torch.cdist(x, y, p=2),
            "cosine": lambda x, y: 1 - cosine_similarity(x[:, None, :], y[None, :, :], dim=-1),
        }

    def fit(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 1024,
        distance_metric: str = "euclidean",
        **kwargs,
    ) -> float:
        """
        Evaluate embeddings using kNN. Distance and topk computations are done in batches.

        :param embeddings: Embedding tensor (n_samples, hidden_size)
        :param labels: Label tensor (n_samples,)
        :param batch_size: Batch size for distance and top-k computation in chunks
        :param distance_metric: Metric to use for distance computation 'euclidean', 'cosine'
        :return: F1-micro score (for classification) or Pearson correlation (for regression)
        """
        num_samples = embeddings.size(0)
        num_classes = len(torch.unique(labels))
        knn_indices = torch.zeros((num_samples, self.k), dtype=torch.long, device=embeddings.device)

        distance = self.distance_metrics.get(distance_metric, self.distance_metrics["euclidean"])

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_features = embeddings[start:end]

            # Distances between the batch and all other features
            distances = distance(batch_features, embeddings)

            # Exclude self-distances by setting diagonal to a large number
            diag_indices = torch.arange(start, end, device=embeddings.device)
            distances[diag_indices - start, diag_indices] = float("inf")

            # Indices of the k nearest neighbors for the batch
            batch_knn_indices = distances.topk(self.k, largest=False).indices
            knn_indices[start:end] = batch_knn_indices

        knn_labels = labels[knn_indices]

        if self.regression:
            # Mean all neighbors for regression
            y_pred = knn_labels.mean(dim=1)
            score = torch.corrcoef(torch.stack([labels, y_pred]))[0, 1].item()
        else:
            # Majority voting for classification
            y_pred = torch.mode(knn_labels, dim=1).values

            task = "multiclass" if num_classes > 2 else "binary"
            score = f1_score(y_pred, labels, task=task, num_classes=num_classes, average="micro").item()

        self.score = score
        return score
