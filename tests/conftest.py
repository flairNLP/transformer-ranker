import pytest
from datasets import Dataset, load_dataset
from transformers import AutoModel


@pytest.fixture(scope="session")
def small_language_models():
    """Use two small models for testing"""
    return (
        AutoModel.from_pretrained("prajjwal1/bert-tiny"),
        AutoModel.from_pretrained("google/electra-small-discriminator"),
    )


@pytest.fixture(scope="session")
def conll():
    """One ner dataset, load once and keep"""
    return load_dataset("conll2003", trust_remote_code=True)


@pytest.fixture(scope="session")
def wnut():
    """One more ner dataset"""
    return load_dataset("wnut_17", trust_remote_code=True)


@pytest.fixture(scope="session")
def trec():
    """One text classification"""
    return load_dataset("trec", trust_remote_code=True)


@pytest.fixture(scope="session")
def sick():
    """Sick has text pairs"""
    return load_dataset("yangwang825/sick", trust_remote_code=True)


@pytest.fixture(scope="session")
def stsb():
    """Sts-b has floats as labels (regression)"""
    return load_dataset("glue", "stsb", trust_remote_code=True)


@pytest.fixture(scope="session")
def custom_dataset():
    """Prepare a small custom dataset"""
    return Dataset.from_dict(
        {
            "text": ["whatsup", "quick", "", "datasets", "test"],
            "text_pair": ["Papers", "with", "code", "humboldt", "focus"],
            "label": [1, 0, 2, 1, 2],
            "extra": [100, 200, 300, 400, 500],
        }
    )
