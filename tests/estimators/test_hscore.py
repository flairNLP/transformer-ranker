import pytest

from transformer_ranker.estimators.hscore import HScore


def test_hscore_on_iris(iris_dataset):
    e = HScore()
    score = e.fit(iris_dataset["data"], iris_dataset["target"])
    assert score == pytest.approx(1.1779776722964175)
