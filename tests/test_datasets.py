import pytest
import torch
from datasets import load_dataset
from transformer_ranker.datacleaner import DatasetCleaner


def test_find_columns(sample_dataset):
    """Test text and label column names"""
    cleaner = DatasetCleaner()
    assert cleaner._find_column(sample_dataset, 'text column') == 'text', "Column name for texts not found"
    assert cleaner._find_column(sample_dataset, 'label column') == 'label', "Column name for labels not found"


def test_merge_text_pairs(sample_dataset):
    """Test merging text pair columns"""
    cleaner = DatasetCleaner(text_column='text', text_pair_column='text_pair')
    merged = cleaner._merge_text_pairs(sample_dataset, 'text', 'text_pair')
    expected = [f"{text} [SEP] {pair}" for text, pair in zip(sample_dataset['text'], sample_dataset['text_pair'])]
    assert merged["text_with_text_pair"] == expected, "Columns are merged incorrectly"


def test_downsample_ratio(sample_dataset):
    """Test dataset size after downsampling"""
    cleaner = DatasetCleaner(dataset_downsample=0.5)
    half_size = cleaner._downsample(sample_dataset, 0.5)
    assert len(half_size) == 2, "Issues with downsampling"


def test_cleanup_rows(sample_dataset):
    """Test empty rows and check size"""
    cleaner = DatasetCleaner()
    cleaner_up = cleaner._cleanup_rows(sample_dataset, 'text', 'label')
    assert len(cleaner_up) == 4, "Should've cleaned out the empty text row."


@pytest.mark.parametrize("task_type,dataset_name,downsample_ratio", [
    ("token classification", "conll2003", 0.01),
    ("token classification", "wnut_17", 0.05),
    ("text classification", "trec", 0.05),
    ("text classification", "stanfordnlp/sst2", 0.005),
    ("text classification", "hate_speech18", 0.025),
    ("text classification", "yangwang825/sick", 0.025),
    ("text classification", "SetFit/rte", 0.05),
])
def test_datacleaner_functionality(task_type, dataset_name, downsample_ratio):
    """Load various datasets and check datacleaner outputs"""
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    cleaner = DatasetCleaner(dataset_downsample=downsample_ratio)
    texts, labels, task_category = cleaner.prepare_dataset(dataset)

    assert task_category == task_type, f"Expected task category '{task_type}' but got '{task_category}'."
    assert isinstance(texts, list) and texts, "Texts should not be empty."
    assert isinstance(labels, torch.Tensor) and labels.numel() > 0, "Labels tensor should not be empty."
    assert (labels >= 0).all(), "Labels should be positive"
