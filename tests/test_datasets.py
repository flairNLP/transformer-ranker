import pytest
import torch
from datasets import Dataset
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

    preprocessor = DatasetCleaner(dataset_downsample=downsampling_ratio)
    dataset = preprocessor.prepare_dataset(dataset_name)

    # Test dataset preprocessing
    assert isinstance(dataset, Dataset), f"Dataset '{dataset_name}' is not a valid Dataset object"

    assert preprocessor.task_type == task_type, (
        f"Task type mismatch: expected '{task_type}', got '{preprocessor.task_type}'"
        f"in dataset '{dataset_name}'"
    )

    # Make sure text and label columns were found
    assert preprocessor.text_column is not None, f"Text column not found in dataset {dataset_name}"
    assert preprocessor.label_column is not None, f"Label column not found in dataset {dataset_name}"

    # Test texts in the text column
    sentences = preprocessor.prepare_sentences(dataset)
    assert isinstance(sentences, list) and len(sentences) > 0, (
        "Sentences/tokens list is empty in dataset %s", dataset_name
    )

    # Ensure the sentences are in the correct format (str for text-classification, List[str] for token-level)
    if task_type == "text classification":
        for sentence in sentences:
            assert isinstance(sentence, str), (
                f"Incorrect sentence type in dataset '{dataset_name}', all expected to be str "
                f"but some sentences have different type ({type(sentence)})."
            )

            # For text and text pair classification, make sure there's no empty strings
            assert sentence != "", f"Empty sentence found in dataset {dataset_name}"

    elif task_type == "token classification":
        for sentence in sentences:
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
    labels = preprocessor.prepare_labels(dataset)
    assert isinstance(labels, torch.Tensor) and labels.size(0) > 0, "Labels tensor is empty"
    assert (labels >= 0).all(), f"Negative label found in dataset {dataset_name}"


def test_simple_dataset():
    original_dataset = Dataset.from_dict({
        "text": ["", "This is a complete sentence.", "b", "c", "d", "e"],
        "label": ["X", "Y", "Z", "X", "Y", "Z"],
        "something_else": [0, 1, 2, 3, 4, 5]
    })

    preprocessor = DatasetCleaner(dataset_downsample=0.5, remove_empty_sentences=False)
    dataset = preprocessor.prepare_dataset(original_dataset)

    assert len(original_dataset) == 6
    assert original_dataset.column_names == ["text", "label", "something_else"]

    assert len(dataset) == 3
    assert dataset.column_names == ["text", "label"]

    preprocessor = DatasetCleaner(remove_empty_sentences=False)
    dataset = preprocessor.prepare_dataset(original_dataset)

    assert dataset["label"] == [0, 1, 2, 0, 1, 2]
    assert original_dataset["label"] == ["X", "Y", "Z", "X", "Y", "Z"]

    preprocessor = DatasetCleaner(remove_empty_sentences=True)
    dataset = preprocessor.prepare_dataset(original_dataset)

    # One row should have been removed in the processed dataset
    assert dataset["label"] == [1, 2, 0, 1, 2]
    assert original_dataset["label"] == ["X", "Y", "Z", "X", "Y", "Z"]
