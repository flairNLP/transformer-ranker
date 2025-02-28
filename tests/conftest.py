import pytest
import torch
from datasets import Dataset, load_dataset
from sklearn import datasets
from transformers import AutoModel


@pytest.fixture(scope="session")
def small_language_models():
    """Use two small models for quick testing."""
    return (
        AutoModel.from_pretrained("prajjwal1/bert-tiny"),
        AutoModel.from_pretrained("google/electra-small-discriminator"),
    )


@pytest.fixture(scope="session")
def sample_dataset():
    """One dummy custom dataset."""
    return Dataset.from_dict(
        {
            "text": ["whatsup", "quick", "", "datasets", "test"],
            "text_pair": ["Python programming", "loading", "student", "humboldt", "focus"],
            "label": [1, 0, 2, 1, 2],
            "extra": [100, 200, 300, 400, 500],
        }
    )


@pytest.fixture(scope="session")
def conll():
    """Ner dataset, load once and keep."""
    return load_dataset("conll2003")


@pytest.fixture(scope="session")
def wnut():
    """One more ner dataset"""
    return load_dataset("wnut_17")


@pytest.fixture(scope="session")
def trec():
    """One text classification"""
    return load_dataset("trec")


@pytest.fixture(scope="session")
def sick():
    """Sick has text pairs"""
    return load_dataset("yangwang825/sick")


@pytest.fixture(scope="session")
def stsb():
    """Sts has floats as labels (regression)"""
    return load_dataset("glue", "stsb")


@pytest.fixture(scope="session")
def iris_dataset():
    iris = datasets.load_iris()
    data = torch.tensor(iris["data"], dtype=torch.float32)
    data[142] += torch.tensor([0, 0, 0, 0.01])  # Ensure no exact duplicates
    return {"data": data, "target": torch.tensor(iris["target"], dtype=torch.float32)}
