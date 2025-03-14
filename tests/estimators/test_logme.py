import pytest

from transformer_ranker.estimators.logme import LogME


def test_logme_iris(iris_dataset):
    """Test LogME with a classification dataset."""
    estimator = LogME()
    score = estimator.fit(iris_dataset["data"], iris_dataset["labels"])
    assert score == pytest.approx(-0.16006022033609904)


def test_logme_california_housing(california_housing_dataset):
    """Test LogME with a regression dataset."""
    estimator = LogME(regression=True)
    score = estimator.fit(california_housing_dataset["data"], california_housing_dataset["labels"])
    assert score == pytest.approx(-1.180475225470693)
