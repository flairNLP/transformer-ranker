import pytest

from transformer_ranker.estimators.logme import LogME


def test_logme_on_iris(iris_dataset):
    e = LogME()
    score = e.fit(iris_dataset["data"], iris_dataset["target"])

    assert score == pytest.approx(-0.16002001310130057)
