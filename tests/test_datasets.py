import pytest
import torch
from datasets import load_dataset

from transformer_ranker.datacleaner import DatasetCleaner, TaskCategory


def test_find_columns(custom_dataset):
    """Test finding text and label columns."""
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
    """Test BIO prefix removal for NER."""
    cleaner = DatasetCleaner()
    conll = conll["train"]
    label_column = cleaner._find_column(conll, "label column")
    _, label_map = cleaner._create_label_map(conll, "ner_tags")
    _, new_label_map = cleaner._remove_bio_encoding(conll, label_column, label_map)
    assert new_label_map == {"O": 0, "PER": 1, "ORG": 2, "LOC": 3, "MISC": 4}


def test_merge_text_pairs(custom_dataset):
    """Test merging of text pairs."""
    cleaner = DatasetCleaner(text_column="text", text_pair_column="other")
    dataset = cleaner._merge_text_pairs(custom_dataset, "text", "other")
    expected = [f"{text} [SEP] {pair}" for text, pair in zip(custom_dataset["text"], custom_dataset["other"])]
    assert dataset["text_pair_other"] == expected, "Columns are merged incorrectly"


@pytest.mark.parametrize(
    "dataset_name,columns,expected_task,expected_dtype,downsample",
    [
        # Text classification
        ("trec", {}, TaskCategory.TEXT_CLASSIFICATION, torch.int64, 0.05),
        ("ag_news", {}, TaskCategory.TEXT_CLASSIFICATION, torch.int64, 0.05),
        ("stanfordnlp/sst2", {}, TaskCategory.TEXT_CLASSIFICATION, torch.int64, 0.05),
        ("mteb/Ddisco", {}, TaskCategory.TEXT_CLASSIFICATION, torch.int64, 0.1),

        # Token classification
        ("conll2003", {}, TaskCategory.TOKEN_CLASSIFICATION, torch.int64, 0.01),
        ("wnut_17", {}, TaskCategory.TOKEN_CLASSIFICATION, torch.int64, 0.01),
        ("benjamin/ner-uk", {}, TaskCategory.TOKEN_CLASSIFICATION, torch.int64, 0.01),
        ("mpsilfve/finer", {}, TaskCategory.TOKEN_CLASSIFICATION, torch.int64, 0.01),

        # Pair classification
        ("yangwang825/sick", {"text_pair_column": "text2"}, TaskCategory.PAIR_CLASSIFICATION, torch.int64, 0.025),
        ("SetFit/rte", {"text_pair_column": "text2"}, TaskCategory.PAIR_CLASSIFICATION, torch.int64, 0.025),
        ("curaihealth/medical_questions_pairs", {"text_pair_column": "question_2"},
        TaskCategory.PAIR_CLASSIFICATION, torch.int64, 0.05),

        # Sentence similarity
        ("SetFit/stsb", {"text_pair_column": "text2"}, TaskCategory.SENTENCE_SIMILARITY, torch.float, 0.05),
        ("mteb/sickr-sts", {"text_pair_column": "sentence2", "label_column": "score"}, TaskCategory.SENTENCE_SIMILARITY, torch.float, 0.05),
        ("Ukhushn/home-depot", {"text_column": "search_term", "text_pair_column": "product_description", "label_column": "relevance"},
        TaskCategory.SENTENCE_SIMILARITY, torch.float, 0.01),
    ]  # fmt: skip
)
def test_dataset_preprocessing(dataset_name, columns, expected_task, expected_dtype, downsample):
    """Test preprocessing across tasks and datasets."""
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    cleaner = DatasetCleaner(dataset_downsample=downsample, **columns)
    texts, labels, task = cleaner.prepare_dataset(dataset)

    assert task == expected_task, f"Expected task '{expected_task}' for {dataset_name}."
    assert isinstance(texts, list) and texts, f"Empty or invalid texts in {dataset_name}."
    assert isinstance(labels, torch.Tensor) and labels.numel() > 0, f"Empty labels in {dataset_name}."
    assert labels.dtype == expected_dtype, f"Expected {expected_dtype} labels for {dataset_name}, got {labels.dtype}."
