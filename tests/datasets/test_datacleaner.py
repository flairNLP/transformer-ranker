import pytest
from datasets import load_dataset, DatasetDict, Dataset
from transformer_ranker.datacleaner import DatasetCleaner
import torch

word_classification_datasets = [
    "conll2003", "wnut_17", "jnlpba", "ncbi_disease",
    # More word classification (flat NER) datasets:
    # "tner/ontonotes5", "levow/msra_ner"
]

sentence_classification_datasets = [
    "trec", "stanfordnlp/sst2", "emotion", "hate_speech18",
    # More text classification datasets:
    # "osanseviero/twitter-airline-sentiment", "SetFit/ade_corpus_v2_classification",
]


def test_load_dataset():
    dataset_names = sentence_classification_datasets + word_classification_datasets

    for dataset in dataset_names:
        try:
            dataset = load_dataset(dataset, trust_remote_code=True)
        except Exception as e:
            pytest.fail(
                "Huggingface loader failed on dataset %s with error: %s" % (dataset, e)
            )


def test_sentence_datasets_datacleaner():
    dataset_names = sentence_classification_datasets

    for dataset in dataset_names:
        preprocessor = DatasetCleaner(merge_data_splits=True)
        dataset = preprocessor.prepare_dataset(dataset)

        # Ensure that the prepared dataset is a Dataset
        assert isinstance(dataset, Dataset)

        # Step 1: Make sure that task type is correctly found
        assert preprocessor.task_type == "sentence classification"

        # Step 2.1: Make sure that text column was found
        assert preprocessor.text_column is not None

        # Step 2.2: Make sure that at least one label column was found
        assert preprocessor.label_column is not None

        # Step 3: Check that prepare_sentences returns a non-empty list of sentences
        sentences = preprocessor.prepare_sentences(dataset)
        assert isinstance(sentences, list) and len(sentences) > 0, (
            "prepare_sentences returned an empty list for dataset %s" % dataset
        )
        assert all(isinstance(sentence, str) for sentence in sentences), (
            "prepare_sentences returned non-string or non-list elements for dataset %s" % dataset
        )

        # Step 4: Check that prepare_labels returns a non-empty torch.Tensor
        labels = preprocessor.prepare_labels(dataset)
        assert isinstance(labels, torch.Tensor) and labels.size(0) > 0, (
            "prepare_labels returned an empty tensor for dataset %s" % dataset
        )


def test_word_datasets_datacleaner():
    dataset_names = word_classification_datasets

    for dataset in dataset_names:
        preprocessor = DatasetCleaner(merge_data_splits=True)
        dataset = preprocessor.prepare_dataset(dataset)

        # Ensure that the prepared dataset is a Dataset or DatasetDict
        assert isinstance(dataset, (DatasetDict, Dataset))

        # Step 1: Make sure that task type is correctly found
        assert preprocessor.task_type == "word classification"

        # Step 2.1: Make sure that text column was found
        assert preprocessor.text_column is not None

        # Step 2.2: Make sure that label column was found
        assert preprocessor.label_column is not None

        # Step 3: Check that prepare_sentences returns a non-empty list of word tokens
        sentences = preprocessor.prepare_sentences(dataset)
        assert isinstance(sentences, list) and len(sentences) > 0, (
            "prepare_sentences returned an empty list for dataset %s" % dataset
        )
        assert all(isinstance(sentence, list) for sentence in sentences), (
            "prepare_sentences returned non-string or non-list elements for dataset %s" % dataset
        )

        # Step 4: Check that prepare_labels returns a non-empty torch.Tensor
        labels = preprocessor.prepare_labels(dataset)
        assert isinstance(labels, torch.Tensor) and labels.size(0) > 0, (
            "prepare_labels returned an empty tensor for dataset %s" % dataset
        )

# pytest test_datacleaner.py -v
