import pytest
from transformer_ranker.estimators.logme import LogME


def test_logme_iris(iris_dataset):
    """Test LogME with a classification dataset."""
    estimator = LogME()
    score = estimator.fit(iris_dataset["data"], iris_dataset["labels"])
    assert score == pytest.approx(-0.16006022033609904)