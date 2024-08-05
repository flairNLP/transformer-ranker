import torch


class LogME:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, regression: bool = False, max_iter: int = 11, tol: float = 1e-3):
        """
        Torch-LogME (Log Marginal Evidence) estimator class.
        Implementation using pytorch to support both cpu and gpu.

        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        paper link: http://proceedings.mlr.press/v139/you21b.html
        original code: https://github.com/thuml/LogME/blob/main/LogME.py

        :param alpha: Initial precision of the prior (controls the regularization strength)
        :param beta: Initial precision of the likelihood (controls the noise in the data)
        :param regression: Boolean flag indicating whether the task is regression.
        :param tol: Tolerance for the optimization convergence
        :param max_iter: Maximum iterations for the optimization process
        """
        self.alpha = alpha
        self.beta = beta
        self.regression = regression
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Measure transferability of features using LogME.

        :param features: Embedding matrix of shape (n_samples, hidden_dim)
        :param labels: Label vector of shape (n_samples,)
        """
        # Center all features
        features = features.to(torch.float64)

        if self.regression:
            labels = labels.to(torch.float64)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(-1)

        # Get the number of samples and the hidden size
        n_samples, hidden_size = features.shape

        # Perform Singular Value Decomposition (SVD) on the features
        u, singular_values, v_transpose = torch.linalg.svd(features, full_matrices=False)

        # Compute sigma which is the square of singular values
        sigma = (singular_values.reshape(-1, 1) ** 2)

        # Initialize the sum of evidences
        evidence_sum = 0.0

        # Determine the number of classes for classification or regression tasks
        num_classes = labels.shape[1] if self.regression else len(torch.unique(labels))

        # Loop over each class (for classification) or each target column (for regression)
        for i in range(num_classes):
            # For classification create a one-hot vector, for regression, use the corresponding column of labels
            labels_ = labels[:, i] if self.regression else (labels == i).to(torch.float64)
            labels_ = labels_.unsqueeze(-1)

            # Project labels onto the singular vectors (x)
            projected_labels = u.T @ labels_
            projected_labels_squared = projected_labels ** 2

            # Compute residual sum of squares. If k < hidden_size, we compute sum of xi for 0 singular values directly
            residual_sum_squares = (labels_ ** 2).sum() - projected_labels_squared.sum()

            # Start with initial alpha and beta values
            alpha, beta = torch.tensor(self.alpha), torch.tensor(self.alpha)

            # To ensure that these variables are defined
            residual_error = torch.tensor(0.0)
            precision_weighted_sum = torch.tensor(0.0)
            tau = torch.tensor(0.0)

            # Iteratively update alpha and beta until convergence or maximum iterations
            for _ in range(self.max_iter):
                tau = alpha / beta  # Ratio of alpha to beta, representing the noise-to-signal ratio
                gamma = (sigma / (sigma + tau)).sum()
                precision_weighted_sum = (sigma * projected_labels_squared / ((tau + sigma) ** 2)).sum()
                residual_error = (projected_labels_squared / ((1 + sigma / tau) ** 2)).sum() + residual_sum_squares

                # Update alpha (prior precision) based on current estimates
                alpha = gamma / (precision_weighted_sum + 1e-5)
                # Update beta (likelihood precision)
                beta = (n_samples - gamma) / (residual_error + 1e-5)

                # Compute the new tau and stop if convergence criterion is met
                new_tau = alpha / beta
                if abs(new_tau - tau) / tau <= self.tol:
                    break

            # Compute evidence based on the optimized alpha and beta
            evidence = (hidden_size / 2.0 * torch.log(alpha)
                        + n_samples / 2.0 * torch.log(beta)
                        - 0.5 * torch.sum(torch.log(alpha + beta * sigma))
                        - beta / 2.0 * residual_error
                        - alpha / 2.0 * precision_weighted_sum
                        - n_samples / 2.0 * torch.log(torch.tensor(2 * torch.pi)))

            # Normalize by the number of samples
            evidence /= n_samples

            # Compute the mean vector for evidence calculation, this can later be used for prediction
            mean_vector = 1.0 / (tau + sigma) * singular_values * projected_labels
            mean_vector = (v_transpose.T @ mean_vector).reshape(-1)

            # Sum the evidence values
            evidence_sum += evidence.item()

        return evidence_sum / num_classes
