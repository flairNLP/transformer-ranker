import pytest
import torch
from datasets import load_dataset
from sklearn import datasets
from transformers import AutoModel


@pytest.fixture(scope="session")
def small_language_models():
    """Use two tiny models for testing"""
    return (
        AutoModel.from_pretrained("prajjwal1/bert-tiny"),
        AutoModel.from_pretrained("google/electra-small-discriminator")
    )


@pytest.fixture(scope="session")
def conll():
    return load_dataset("conll2003")


@pytest.fixture(scope="session")
def trec():
    return load_dataset("trec")


@pytest.fixture(scope="session")
def iris_dataset():
    iris = datasets.load_iris()
    data = torch.tensor(iris["data"], dtype=torch.float32)
    data[142] += torch.tensor([0, 0, 0, 0.01])  # Ensure no exact duplicates
    return {
        "data": data,
        "target": torch.tensor(iris["target"], dtype=torch.float32)
    }
