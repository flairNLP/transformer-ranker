import pytest

import torch


@pytest.fixture(scope="session")
def iris_dataset():
    from sklearn import datasets

    iris = datasets.load_iris()


    data = torch.tensor(iris["data"], dtype=torch.float32)
    data[142] += torch.tensor([0, 0, 0, 0.01])  # make duplicate element unique

    return {
        "data": data,
        "target": torch.tensor(iris["target"], dtype=torch.float32)
    }
