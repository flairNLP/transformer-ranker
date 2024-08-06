from typing import Tuple

import pytest
import torch

from transformer_ranker.estimators import KNN


def sample_data(
    k: int = 6, dim: int = 1024, distance: float = 1.0, radius: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, float]:

    num_correct = (k // 2) + 2  # +1 for majority, +1 for the datapoint itself
    num_incorrect = (k - 1) // 2  # these will be the minority

    print(k, num_correct, num_incorrect)

    num_total = 2 * (num_correct + num_incorrect)

    # Generate a three-class dataset with the datapoints on two spheres
    data = torch.nn.functional.normalize(torch.rand(num_total, dim), dim=1) * radius

    labels = torch.tensor(
        [0] * num_correct
        + [2] * num_incorrect
        + [1] * num_correct
        + [2] * num_incorrect
    )

    diff = torch.nn.functional.normalize(torch.rand(dim), dim=0) * distance

    data += torch.rand(dim)
    data[num_correct + num_incorrect :] += diff

    expected_accuracy = (num_correct) / (num_correct + num_incorrect)

    return data, labels, expected_accuracy


@pytest.mark.parametrize("k,dim", [(6, 1024), (10, 100), (100, 256), (1024, 16)])
def test_knn_on_constructed_data(k, dim):
    features, labels, expected_accuracy = sample_data(k=k, dim=dim)
    estimator = KNN(k)

    accuracy = estimator.fit(features, labels)

    assert accuracy == pytest.approx(expected_accuracy)


@pytest.mark.parametrize(
    "k, expected_accuracy",
    [
        (1, 0.96),
        (3, 0.96),
        (5, 0.9666666666666667),
        (13, 0.9666666666666667),
    ],
)
def test_knn_on_iris(iris_dataset, k, expected_accuracy):
    e = KNN(k)
    score = e.fit(iris_dataset["data"], iris_dataset["target"])
    assert score == pytest.approx(expected_accuracy)
