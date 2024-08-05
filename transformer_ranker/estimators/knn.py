import torch
from torchmetrics import F1Score


class KNN:
    def __init__(
        self,
        k: int = 3,
        regression: bool = False,
    ):
        """
        Initializes the k-Nearest Neighbor estimator class.
        Implementation using pytorch to support both cpu and gpu.

        :param k: Number of nearest neighbors to consider.
        :param regression: Boolean flag indicating whether the task is regression.
        """
        self.k = k
        self.regression = regression
        self.score = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, batch_size: int = 1024) -> float:
        """
        Evaluate features using k-Nearest Neighbours
        (i.e. similar to leave-one-out cross validation)

        :param features: Embedding matrix of shape (n_samples, hidden_size)
        :param labels: Label vector of shape (n_samples,)
        :param batch_size: Batch size for processing distance and top-k computation in chunks
        :return: Score (F1 score for classification or Pearson correlation for regression)
        """
        num_samples = features.size(0)
        knn_indices = torch.zeros((num_samples, self.k), dtype=torch.long, device=features.device)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_features = features[start:end]

            # Compute Euclidean distances between the batch and all other features
            dists = torch.cdist(batch_features, features, p=2)

            # Exclude the feature itself by setting diagonal to a large number
            diag_indices = torch.arange(start, end, device=features.device)
            dists[diag_indices - start, diag_indices] = float('inf')

            # Get the indices of the k nearest neighbors for the batch
            batch_knn_indices = dists.topk(self.k, largest=False).indices
            knn_indices[start:end] = batch_knn_indices

        # Retrieve the labels of the k nearest neighbors
        knn_labels = labels[knn_indices]

        if self.regression:
            # Mean continuous values of each neighbor to get prediction
            y_pred = knn_labels.mean(dim=1)

            # Use pearson correlation as a metric for regression tasks
            score = torch.corrcoef(torch.stack([labels, y_pred]))[0, 1].item()
        else:
            # Perform majority voting to predict the label
            y_pred = torch.mode(knn_labels, dim=1).values

            f1 = F1Score(average='micro', num_classes=len(torch.unique(labels)))
            score = f1(y_pred, labels).item()

        self.score = score
        return score
