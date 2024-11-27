import pytest
import torch
from datasets import Dataset, load_dataset
from transformer_ranker.datacleaner import DatasetCleaner


test_datasets = [
    # Token datasets
    ("token classification", "conll2003", 0.01),
    ("token classification", "wnut_17", 0.05),

    # Text classification
    ("text classification", "trec", 0.05),
    ("text classification", "stanfordnlp/sst2", 0.005),
    ("text classification", "hate_speech18", 0.025),

    # Text-pair classification
    ("text classification", "yangwang825/sick", 0.025),
    ("text classification", "SetFit/rte", 0.05),
]


@pytest.mark.parametrize("task_type,dataset_name,downsampling_ratio", test_datasets)
def test_datacleaner(task_type, dataset_name, downsampling_ratio):

    dataset = load_dataset(dataset_name, trust_remote_code=True)
    datacleaner = DatasetCleaner(dataset_downsample=downsampling_ratio)
    texts, labels, task_category = datacleaner.prepare_dataset(dataset)

    assert task_category == task_type, (
        f"Task type mismatch: expected '{task_type}', got '{task_category}'"
        f"in dataset '{dataset_name}'"
    )

    assert isinstance(texts, list) and len(texts) > 0, (
        "Sentences/tokens list is empty in dataset %s", dataset_name
    )

    # Ensure the sentences are in the correct format
    # (str for text-classification, List[str] for token-level)
    if task_type == "text classification":
        for sentence in texts:
            assert isinstance(sentence, str), (
                f"Incorrect sentence type in dataset '{dataset_name}', all expected to be str "
                f"but some sentences have different type ({type(sentence)})."
            )

            # For text and text pair classification, make sure there's no empty strings
            assert sentence != "", f"Empty sentence found in dataset {dataset_name}"

    elif task_type == "token classification":
        for sentence in texts:
            # For token classification, make sure there is no empty lists of tokens
            assert len(sentence) >= 0,  f"Empty token list found in dataset {dataset_name}"

            # Check that no empty strings exist within the token lists
            for token in sentence:
                assert isinstance(token, str), f"Non-str token ({type(token)}) found within a sentence in dataset {dataset_name}."
                assert len(token) >= 0, f"Empty token found within a sentence in dataset {dataset_name}."

    else:
        msg = f"Uncrecognized task type '{task_type}'."
        raise KeyError(msg)

    # Test the label column in each dataset
    assert isinstance(labels, torch.Tensor) and labels.size(0) > 0, "Labels tensor is empty"
    assert (labels >= 0).all(), f"Negative label found in dataset {dataset_name}"


def test_simple_dataset():
    original_dataset = Dataset.from_dict({
        "text": ["", "This is a complete sentence.", "b", "c", "d", "e"],
        "label": ["X", "Y", "Z", "X", "Y", "Z"],
        "something_else": [0, 1, 2, 3, 4, 5]
    })

    preprocessor = DatasetCleaner(dataset_downsample=0.5, cleanup_rows=False)
    texts, labels, task_category = preprocessor.prepare_dataset(original_dataset)

    assert len(original_dataset) == 6
    assert len(texts) == len(labels) == 3  # after downsampling

    preprocessor = DatasetCleaner(cleanup_rows=False)
    texts, labels, task_category = preprocessor.prepare_dataset(original_dataset)

    assert original_dataset["label"] == ["X", "Y", "Z", "X", "Y", "Z"]
    assert torch.equal(labels, torch.tensor([0, 1, 2, 0, 1, 2]))

    preprocessor = DatasetCleaner(cleanup_rows=True)
    texts, labels, task_category = preprocessor.prepare_dataset(original_dataset)

    # One row should have been removed in the processed dataset
    assert original_dataset["label"] == ["X", "Y", "Z", "X", "Y", "Z"]
    assert torch.equal(labels, torch.tensor([1, 2, 0, 1, 2]))
