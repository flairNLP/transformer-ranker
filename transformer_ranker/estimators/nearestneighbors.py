from typing import Union, Optional
import torch
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score


class KNN:
    def __init__(
        self,
        k: int = 3,
        regression: bool = False,
    ):
        """
        K-Nearest Neighbors estimator.

        :param k: Number of nearest neighbors to consider.
        :param regression: Boolean flag if the task is regression.
        """
        self.k = k
        self.regression = regression
        self.score: Optional[float] = None

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor, batch_size: int = 1024) -> float:
        """
        Estimate embedding suitability for classification or regression using nearest neighbors

        :param embeddings: Embedding matrix of shape (n_samples, hidden_size)
        :param labels: Label vector of shape (n_samples,)
        :param batch_size: Batch size for distance and top-k computation in chunks
        :return: Score (F1 score for classification or Pearson correlation for regression)
        """
        num_samples = embeddings.size(0)
        num_classes = len(torch.unique(labels))
        knn_indices = torch.zeros((num_samples, self.k), dtype=torch.long, device=embeddings.device)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_features = embeddings[start:end]

            # Euclidean distances between the batch and all other features
            dists = torch.cdist(batch_features, embeddings, p=2)

            # Exclude self-distances by setting diagonal to a large number
            diag_indices = torch.arange(start, end, device=embeddings.device)
            dists[diag_indices - start, diag_indices] = float("inf")

            # Indices of the k nearest neighbors for the batch
            batch_knn_indices = dists.topk(self.k, largest=False).indices
            knn_indices[start:end] = batch_knn_indices

        knn_labels = labels[knn_indices]

        if self.regression:
            # Mean all neighbors for regression
            y_pred = knn_labels.mean(dim=1)
            score = torch.corrcoef(torch.stack([labels, y_pred]))[0, 1].item()
        else:
            # Majority voting for classification
            y_pred = torch.mode(knn_labels, dim=1).values

            f1: Union[BinaryF1Score, MulticlassF1Score]
            if num_classes == 2:
                f1 = BinaryF1Score(average="micro")
            else:
                f1 = MulticlassF1Score(average="micro", num_classes=num_classes)

            score = f1(y_pred, labels).item()

        self.score = score
        return score
