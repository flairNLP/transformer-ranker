import torch

from .base import Estimator


class LogME(Estimator):
    def __init__(self, regression: bool = False):
        """
        Log of Maximum Evidence
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
        labels = labels.to(torch.float64)

        num_samples, hidden_size = embeddings.shape
        class_names = torch.unique(labels) if not self.regression else None
        num_classes = len(class_names) if not self.regression else 1

        # Decompose embeddings via SVD
        u, s, _ = torch.linalg.svd(embeddings, full_matrices=False)
        sigma = (s**2).unsqueeze(-1)

        alpha, beta = torch.tensor(initial_alpha), torch.tensor(initial_beta)
        evidence_sum = 0.0

        for i in range(num_classes):
            # Use one-hot vectors for classification
            labels_ = labels if self.regression else (labels == class_names[i]).to(dtype=torch.float64)
            labels_ = labels_.unsqueeze(-1).to(embeddings.device)

            # Project labels to singular vectors (x)
            projected_labels = (u.T @ labels_) ** 2
            residual_sum_squares = (labels_**2).sum() - projected_labels.sum()

            residual_error = torch.tensor(0.0)
            precision_weighted_sum = torch.tensor(0.0)

            # Iteratively update alpha and beta until convergence or max_iter
            for _ in range(max_iter):
                tau = alpha / beta
                gamma = (sigma / (sigma + tau)).sum()
                precision_weighted_sum = (sigma * projected_labels / ((tau + sigma) ** 2)).sum()
                residual_error = (projected_labels / ((1 + sigma / tau) ** 2)).sum() + residual_sum_squares

                # Update alpha and beta
                alpha = gamma / (precision_weighted_sum + 1e-5)
                beta = (num_samples - gamma) / (residual_error + 1e-5)
                if abs(alpha / beta - tau) / tau <= tol:
                    break

            # Compute model evidence
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
