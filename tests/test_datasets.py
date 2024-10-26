from typing import List, Tuple, Type, Union

import pytest
import torch
from datasets import Dataset
from transformer_ranker.datacleaner import DatasetCleaner


def load_datasets(dataset_type: str, num_datasets: Union[str, int] = "all") -> Tuple[List[str], Type, Type]:
    """Try loading and preparing different datasets"""
    dataset_map = {
        'token': (
            ["conll2003", "wnut_17"],
            "token classification", list
        ),
        'text': (
            ["trec", "stanfordnlp/sst2", "hate_speech18"],
            "text classification", str
        ),
        'text_pair': (
            ["yangwang825/sick", "SetFit/rte"],
            "text classification", str
        )
    }

    datasets, task_type, sentence_type = dataset_map[dataset_type]
    if isinstance(num_datasets, int):
        datasets = datasets[:num_datasets]

    return datasets, task_type, sentence_type


def validate_dataset(
        preprocessor,
        dataset_name: str,
        dataset: Dataset,
        expected_task_type: Type,
        sentence_type: Type
):
    assert isinstance(dataset, Dataset), f"Dataset '{dataset_name}' is not a valid Dataset object"

    assert preprocessor.task_type == expected_task_type, \
        (f"Task type mismatch: expected '{expected_task_type}', got '{preprocessor.task_type}'"
         f"in dataset '{dataset_name}'")

    # Make sure text and label columns were found
    assert preprocessor.text_column is not None, f"Text column not found in dataset {dataset_name}"
    assert preprocessor.label_column is not None, f"Label column not found in dataset {dataset_name}"

    # Test texts in the text column
    sentences = preprocessor.prepare_sentences(dataset)
    assert isinstance(sentences, list) and len(sentences) > 0, (
        "Sentences/tokens list is empty in dataset %s", dataset_name
    )
    assert all(isinstance(sentence, sentence_type) for sentence in sentences), \
        (f"Incorrect sentence/token type in dataset '{dataset_name}', all expected to be '{sentence_type}' "
         f"but some sentences have different type")

    if sentence_type == str:
        # For text and text pair classification, make sure there's no empty strings
        assert all(sentence != "" for sentence in sentences), f"Empty sentence found in dataset {dataset_name}"

    if sentence_type == list:
        # For token classification, make sure there is no empty tokens
        assert all(sentence != [] for sentence in sentences), f"Empty token list found in dataset {dataset_name}"
        # Check that no empty strings exist within the token lists
        assert all(all(token != "" for token in sentence) for sentence in sentences), \
            f"Empty token found within a sentence in dataset {dataset_name}"

    # Test the label column in each dataset
    labels = preprocessor.prepare_labels(dataset)
    assert isinstance(labels, torch.Tensor) and labels.size(0) > 0, "Labels tensor is empty"
    assert (labels >= 0).all(), f"Negative label found in dataset {dataset_name}"


@pytest.mark.parametrize("dataset_type", ["text", "token", "text_pair"])
def test_datacleaner(dataset_type):
    datasets, task_type, sentence_type = load_datasets(dataset_type, "all")

    # Loop through all test datasets, down sample them to 0.2
    for dataset_name in datasets:
        preprocessor = DatasetCleaner(dataset_downsample=0.2)
        dataset = preprocessor.prepare_dataset(dataset_name)

        # Test dataset preprocessing
        validate_dataset(preprocessor, dataset_name, dataset, task_type, sentence_type)
