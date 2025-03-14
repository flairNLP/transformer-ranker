import pytest
from conftest import generate_sample_dataset

from transformer_ranker.estimators import NearestNeighbors


@pytest.mark.parametrize("k,dim", [(6, 1024), (10, 100), (100, 256), (1024, 16)])
def test_knn_on_constructed_data(k, dim):
    features, labels, expected_accuracy = generate_sample_dataset(k=k, dim=dim)
    estimator = NearestNeighbors(k=k)
    accuracy = estimator.fit(features, labels)
    assert accuracy == pytest.approx(expected_accuracy)


@pytest.mark.parametrize(
    "k, expected_accuracy",
    [(1, 0.96), (3, 0.96), (5, 0.9666666666666667), (13, 0.9666666666666667)],
)
def test_knn_iris(iris_dataset, k, expected_accuracy):
    """Test nearest neighbors on a classification dataset."""
    estimator = NearestNeighbors(k=k)
    score = estimator.fit(iris_dataset["data"], iris_dataset["labels"])
    assert score == pytest.approx(expected_accuracy)


@pytest.mark.parametrize("k", [1, 3, 5, 13])
def test_knn_california_housing(california_housing_dataset, k):
    """Test nearest neighbors on a regression dataset."""
    estimator = NearestNeighbors(k=k, regression=True)
    score = estimator.fit(california_housing_dataset["data"], california_housing_dataset["labels"])
    assert score > 0.5
