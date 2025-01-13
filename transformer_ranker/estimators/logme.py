import torch

from .base import Estimator


class LogME(Estimator):
    def __init__(self, regression: bool = False):
        """
        LogME (Log of Maximum Evidence) estimator.
        Paper: https://arxiv.org/abs/2102.11005
        """
        super().__init__(regression=regression)

    def fit(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        max_iter: int = 11,
        tol: float = 1e-3,
        **kwargs,
    ) -> float:
        """
        LogME intuition: estimate the evidence for embeddings by iteratively optimizing
        the prior (alpha) and likelihood (beta), projecting the target labels onto the singular
        vectors of the feature matrix.

        :param embeddings: Embedding tensor (num_samples, hidden_dim)
        :param labels: Label tensor (num_samples,)
        :param initial_alpha: Initial precision of the prior (controls the regularization strength)
        :param initial_beta: Initial precision of the likelihood (controls the noise in the data)
        :param tol: Tolerance for the optimization convergence
        :param max_iter: Maximum iterations to optimize alpha and beta
        :return: LogME score
        """
        embeddings = embeddings.to(torch.float64)
        labels = labels.to(torch.float64).unsqueeze(-1) if self.regression and labels.dim() == 1 else labels

        # Get the number of samples, number of classes, and the hidden size
        num_samples, hidden_size = embeddings.shape
        class_names, _ = torch.unique(labels, return_counts=True)
        num_classes = labels.shape[1] if self.regression else len(class_names)

        # SVD on the features
        u, singular_values, _ = torch.linalg.svd(embeddings, full_matrices=False)

        # Compute sigma which is the square of singular values
        sigma = singular_values.reshape(-1, 1) ** 2

        evidence_sum = 0.0

        # Start with initial alpha and beta values
        alpha, beta = torch.tensor(initial_alpha), torch.tensor(initial_beta)

        # Loop over each class (for classification) or each target column (for regression)
        for i in range(num_classes):
            # Use one-hot vectors for classification and label columns for regression
            labels_ = labels[:, i] if self.regression else (labels == class_names[i]).to(torch.float64)
            labels_ = labels_.unsqueeze(-1)

            # Project labels onto the singular vectors (x)
            projected_labels = u.T @ labels_
            projected_labels_squared = projected_labels**2

            # Residual sum of squares; if k < hidden_size, sum xi for zero singular values
            residual_sum_squares = (labels_**2).sum() - projected_labels_squared.sum()

            residual_error = torch.tensor(0.0)
            precision_weighted_sum = torch.tensor(0.0)

            # Iteratively update alpha and beta until convergence or maximum iterations
            for _ in range(max_iter):
                tau = alpha / beta  # Alpha-to-beta ratio, representing noise-to-signal
                gamma = (sigma / (sigma + tau)).sum()
                precision_weighted_sum = (sigma * projected_labels_squared / ((tau + sigma) ** 2)).sum()
                residual_error = (projected_labels_squared / ((1 + sigma / tau) ** 2)).sum() + residual_sum_squares

                # Update alpha (prior precision) and beta (likelihood precision)
                alpha = gamma / (precision_weighted_sum + 1e-5)
                beta = (num_samples - gamma) / (residual_error + 1e-5)

                # Compute the new tau and stop if convergence criterion is met
                new_tau = alpha / beta
                if abs(new_tau - tau) / tau <= tol:
                    break

            # Compute evidence using optimized alpha and beta
            evidence = (
                hidden_size / 2.0 * torch.log(alpha)
                + num_samples / 2.0 * torch.log(beta)
                - 0.5 * torch.sum(torch.log(alpha + beta * sigma))
                - beta / 2.0 * residual_error
                - alpha / 2.0 * precision_weighted_sum
                - num_samples / 2.0 * torch.log(torch.tensor(2 * torch.pi))
            )
            evidence /= num_samples

            # Sum the evidence for each class
            evidence_sum += evidence.item()

        logme_score = evidence_sum / num_classes
        self.score = logme_score

        return logme_score
