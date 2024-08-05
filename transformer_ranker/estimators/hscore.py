import torch


class HScore:
    def __init__(self):
        """
        Regularized H-Score estimator class.
        Implementation using pytorch to support both cpu and gpu.

        Original H-score paper:
        And the regularized version: `Newer is not always better: Rethinking transferability metrics, their
        peculiarities, stability and performance (NeurIPS 2021) <https://openreview.net/pdf?id=iz_Wwmfquno>`.
        """
        # Store the computed h-score after fitting
        self.score = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Measure transferability of features using Regularized H-Score.

        :param features: Feature (i.e. embedding) matrix of shape (n_samples, hidden_size)
        :param labels: Label vector of shape (n_samples,)
        :return: Regularized H-Score
        """
        # Center all features
        features = features.to(torch.float64)
        features = features - features.mean(dim=0, keepdim=True)

        # Get the number of samples, the hidden size (i.e. embedding length), and number of classes
        n_samples, hidden_size = features.size()
        n_classes = torch.unique(labels).size(0)

        # Compute the covariance matrix (hidden_size x hidden_size)
        cov_matrix = torch.mm(features.T, features) / n_samples

        # Compute the Ledoit-Wolf shrinkage
        squared_features = features ** 2
        emp_cov_trace = torch.sum(squared_features, dim=0) / n_samples
        mean_cov = torch.sum(emp_cov_trace) / hidden_size

        beta_ = torch.sum(torch.mm(squared_features.T, squared_features)) / n_samples
        delta_ = torch.sum(cov_matrix ** 2)

        beta = (beta_ - delta_) / (hidden_size * n_samples)
        delta = delta_ - 2.0 * mean_cov * emp_cov_trace.sum() + hidden_size * mean_cov**2
        delta /= hidden_size

        # Prevent shrinking more than 1, which would invert the value of covariances
        shrinkage = torch.clamp(beta / delta, 0, 1)

        identity_matrix = torch.eye(features.size(1), device=features.device)

        # Apply the Ledoit-Wolf shrinkage, regularized covariance matrix
        covf_alpha = (1 - shrinkage) * cov_matrix + shrinkage * mean_cov * identity_matrix

        # Matrix of conditional expectations (n_samples, hidden_dim)
        g = torch.zeros(n_classes, hidden_size, dtype=torch.float64, device=features.device)
        for i in range(n_classes):
            mask = labels == i
            Ef_i = features[mask].mean(dim=0)
            g[i] = Ef_i * torch.sqrt(mask.sum())

        # Compute the covariance matrix of g (n_samples, hidden_size)
        covg = torch.mm(g.T, g) / (n_samples - 1)

        # Pseudo-inverse of the regularized covariance matrix
        pinv_covf_alpha = torch.linalg.pinv(covf_alpha, rcond=1e-15)

        # The regularized H-score, a scalar value
        hscore = torch.trace(torch.mm(pinv_covf_alpha, (1 - shrinkage) * covg)).item()
        self.score = hscore

        return hscore
