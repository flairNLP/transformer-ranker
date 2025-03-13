import pytest
from transformer_ranker.estimators.hscore import HScore


def test_hscore_iris(iris_dataset):
    """Test H-Score on a classification dataset."""
    estimator = HScore()
    score = estimator.fit(iris_dataset["data"], iris_dataset["labels"])
    assert score == pytest.approx(1.177842480406727)


def test_hscore_california_housing(california_housing_dataset):
    """H-Score should not be suited for regression."""
    with pytest.raises(ValueError, match=r"HScore is not suitable for regression"):
        estimator = HScore(regression=True)
        score = estimator.fit(california_housing_dataset["data"], california_housing_dataset["labels"])
