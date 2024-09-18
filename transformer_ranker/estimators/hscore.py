import torch


class HScore:
    def __init__(self):
        """
        Regularized H-Score estimator.
        Original H-score paper: https://arxiv.org/abs/2212.10082
        Improved, shrinkage-based H-Score: https://openreview.net/pdf?id=iz_Wwmfquno
        """
        self.score = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """
        H-score intuition: Higher variance between features of different classes (mean vectors for each class)
        and lower feature redundancy (i.e. inverse of the covariance matrix for all data points)
        lead to better transferability.

        :param features: Feature (i.e. embedding) matrix of shape (num_samples, hidden_size)
        :param labels: Label vector of shape (num_samples,)
        :return: H-score, where higher is better.
        """
        # Center all features
        features = features.to(torch.float64)
        features = features - features.mean(dim=0, keepdim=True)

        # Number of samples, hidden size (i.e. embedding length), number of classes
        num_samples, hidden_size = features.size()
        num_classes = torch.unique(labels).size(0)

        # Feature covariance matrix (hidden_size x hidden_size)
        cov_matrix = torch.mm(features.T, features) / num_samples

        # Compute beta and delta for the Ledoit-Wolf shrinkage
        squared_features = features ** 2
        emp_cov_trace = torch.sum(squared_features, dim=0) / num_samples
        mean_cov = torch.sum(emp_cov_trace) / hidden_size
        beta_ = torch.sum(torch.mm(squared_features.T, squared_features)) / num_samples
        delta_ = torch.sum(cov_matrix ** 2)
        beta = (beta_ - delta_) / (hidden_size * num_samples)
        delta = delta_ - 2.0 * mean_cov * emp_cov_trace.sum() + hidden_size * mean_cov**2
        delta /= hidden_size

        # Apply Ledoit-Wolf shrinkage to the feature covariance matrix
        shrinkage = torch.clamp(beta / delta, 0, 1)
        identity_matrix = torch.eye(features.size(1), device=features.device)
        covf_alpha = (1 - shrinkage) * cov_matrix + shrinkage * mean_cov * identity_matrix

        # Pseudo-inverse of the feature covariance matrix
        pinv_covf_alpha = torch.linalg.pinv(covf_alpha, rcond=1e-15)

        # Matrix of class-conditioned means
        class_means = torch.zeros(num_classes, hidden_size, dtype=torch.float64, device=features.device)
        for i in range(num_classes):
            mask = labels == i
            class_features = features[mask].mean(dim=0)
            class_means[i] = class_features * torch.sqrt(mask.sum())

        # Covariance for class-conditioned means
        covg = torch.mm(class_means.T, class_means) / num_samples

        # Shrinkage-based H-score
        hscore = torch.trace(torch.mm(pinv_covf_alpha, (1 - shrinkage) * covg)).item()
        self.score = hscore

        return hscore
