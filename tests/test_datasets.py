import pytest
import torch
from datasets import load_dataset

from transformer_ranker.datacleaner import DatasetCleaner, TaskCategory


def test_find_columns(custom_dataset):
    """Test that text and label column names can be found."""
    cleaner = DatasetCleaner()
    assert cleaner._find_column(custom_dataset, "text column") == "text"
    assert cleaner._find_column(custom_dataset, "label column") == "label"


def test_downsample_ratio(custom_dataset):
    """Test size after downsampling."""
    cleaner = DatasetCleaner(dataset_downsample=0.5)
    dataset = cleaner._downsample(custom_dataset, 0.5)
    assert len(dataset) == 2, "Issues with downsampling"


def test_cleanup_rows(custom_dataset):
    """Test removing empty rows."""
    cleaner = DatasetCleaner()
    dataset = cleaner._cleanup_rows(custom_dataset, "text", "label")
    assert len(dataset) == 4, "Empty texts should've been removed"


def test_label_map(trec, conll):
    """Test label maps for trec and conll."""
    cleaner = DatasetCleaner()
    trec, conll = trec["train"], conll["train"]

    trec_map = {"ABBR": 0, "ENTY": 1, "DESC": 2, "HUM": 3, "LOC": 4, "NUM": 5}
    _, label_map = cleaner._create_label_map(trec, "coarse_label")
    assert label_map == trec_map

    conll_map = {
        "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8,
    }  # fmt: skip
    _, label_map = cleaner._create_label_map(conll, "ner_tags")
    assert label_map == conll_map


def test_bio_decoding(conll):
    """Test removing of BIO prefixes for NER labels."""
    cleaner = DatasetCleaner()
    conll = conll["train"]
    label_column = cleaner._find_column(conll, "label column")
    _, label_map = cleaner._create_label_map(conll, "ner_tags")
    _, new_label_map = cleaner._remove_bio_encoding(conll, label_column, label_map)
    assert new_label_map == {"O": 0, "PER": 1, "ORG": 2, "LOC": 3, "MISC": 4}


def test_merge_text_pairs(custom_dataset):
    """Test merging text pair columns."""
    cleaner = DatasetCleaner(text_column="text", text_pair_column="text_pair")
    dataset = cleaner._merge_text_pairs(custom_dataset, "text", "text_pair")
    expected = [f"{text} [SEP] {pair}" for text, pair in zip(custom_dataset["text"], custom_dataset["text_pair"])]
    assert dataset["text_with_text_pair"] == expected, "Columns are merged incorrectly"


@pytest.mark.parametrize(
    "dataset_name,text_pair_column,task_category",
    [
        ("trec", None, TaskCategory.TEXT_CLASSIFICATION),
        ("yangwang825/sick", "text2", TaskCategory.TEXT_PAIR_CLASSIFICATION),
        ("SetFit/rte", "text2", TaskCategory.TEXT_PAIR_CLASSIFICATION),
        ("SetFit/stsb", "text2", TaskCategory.TEXT_PAIR_REGRESSION),
    ],
)
def test_task_category_assignment(dataset_name, text_pair_column, task_category):
    """Test task category assignment for different datasets."""
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    cleaner = DatasetCleaner(text_pair_column=text_pair_column, dataset_downsample=0.05)
    _, _, task = cleaner.prepare_dataset(dataset)
    assert task == task_category, f"Expected '{task_category}',got '{task}' for {dataset_name}."


@pytest.mark.parametrize(
    "dataset_name,task_category,downsample_ratio",
    [
        ("conll2003", TaskCategory.TOKEN_CLASSIFICATION, 0.01),
        ("wnut_17", TaskCategory.TOKEN_CLASSIFICATION, 0.05),
        ("trec", TaskCategory.TEXT_CLASSIFICATION, 0.05),
        ("stanfordnlp/sst2", TaskCategory.TEXT_CLASSIFICATION, 0.005),
        ("hate_speech18", TaskCategory.TEXT_CLASSIFICATION, 0.025),
        ("yangwang825/sick", TaskCategory.TEXT_CLASSIFICATION, 0.025),
        ("SetFit/rte", TaskCategory.TEXT_CLASSIFICATION, 0.05),
        ("SetFit/stsb", TaskCategory.TEXT_REGRESSION, 0.05),
    ],
)
def test_different_datasets(dataset_name, task_category, downsample_ratio):
    """Load different datasets and verify preprocessing of texts, labels, and task category."""
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    cleaner = DatasetCleaner(dataset_downsample=downsample_ratio)
    texts, labels, task = cleaner.prepare_dataset(dataset)

    assert task == task_category, f"Expected {task_category}, got {task} for {dataset_name}."
    assert isinstance(texts, list) and texts, f"Empty texts in {dataset_name}."
    assert isinstance(labels, torch.Tensor) and labels.numel() > 0, f"Empty labels in {dataset_name}."
