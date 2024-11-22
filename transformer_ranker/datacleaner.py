import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import datasets
import torch
from datasets.dataset_dict import Dataset, DatasetDict
from tokenizers.pre_tokenizers import Whitespace

from .utils import configure_logger

logger = configure_logger("transformer_ranker", logging.INFO)


class TaskCategory(str, Enum):
    """Supported tasks"""
    TEXT_REGRESSION = "text regression"
    TEXT_CLASSIFICATION = "text classification"
    TOKEN_CLASSIFICATION = "token classification"

    def __str__(self):
        return self.value


class PreprocessedDataset(Dataset):
    """A preprocessed dataset with only the required columns (texts and labels),
    down-sampled and cleaned. Provides easy access to texts (sentences/words), labels (tensors),
    and the task category (classification or regression)."""
    def __init__(
        self,
        dataset: Dataset,
        text_column: str,
        label_column: str,
        task_category: TaskCategory,
    ):
        super().__init__(dataset.data, dataset.info)

        self.text_column = text_column
        self.label_column = label_column
        self.task_category = task_category

    def texts(self) -> list[str]:
        """Gather all texts from the text column."""
        return self[self.text_column]

    def labels(self) -> torch.Tensor:
        """Prepare labels as tensors."""
        if self.task_category == TaskCategory.TOKEN_CLASSIFICATION:
            labels = [word_label for labels in self[self.label_column] for word_label in labels]
        else:
            labels = self[self.label_column]
        return torch.tensor(labels)


@dataclass
class DatasetCleaner:
    dataset_downsample: Optional[float] = None
    text_column: Optional[str] = None
    text_pair_column: Optional[str] = None
    label_column: Optional[str] = None
    label_map: Optional[dict] = None
    task_type: Optional[TaskCategory] = None
    cleanup_rows: bool = True
    convert_bio_encoding: bool = True
    tokenize: bool = False

    def prepare_dataset(self, dataset: Union[str, Dataset, DatasetDict]) -> PreprocessedDataset:
        """Clean and verify the dataset by finding text and label fields, task type,
        removing invalid entries, mapping labels, and down-sampling."""

        # Verify a dataset
        if not isinstance(dataset, (Dataset, DatasetDict)):
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

        # Merge splits into one
        dataset = self.merge_dataset_splits(dataset)

        # Search for text and label columns
        text_column = self.text_column if self.text_column \
            else self.find_column("Text column", dataset)
        label_column = self.label_column if self.label_column \
            else self.find_column("Label column", dataset)

        # Concat columns for text pairs
        if self.text_pair_column:
            dataset = self.merge_text_pairs(text_column, self.text_pair_column, dataset)
            text_column = f"{text_column}+{self.text_pair_column}"

        # Assign task category
        task_category = self.task_type if self.task_type \
            else self.find_task_category(label_column, dataset)

        # Remove unused columns
        dataset = dataset.select_columns([text_column, label_column])

        if self.dataset_downsample:
            dataset = self.downsample(self.dataset_downsample, dataset)

        # Remove empty sentences and unsupported labels
        if self.cleanup_rows:
            dataset = self.remove_empty_rows(text_column, label_column, dataset)

        # Optional tokenization if texts are not already tokenized
        if self.tokenize and isinstance(dataset[text_column][0], str):
            dataset = self.whitespace_tokenize(text_column, dataset)

        # Create the label map
        label_map = self.label_map if self.label_map \
            else self.create_label_map(label_column, dataset)

        if isinstance(dataset[label_column][0], str):
            dataset = self.make_labels_categorical(label_column, label_map, dataset)

        if self.convert_bio_encoding and task_category == TaskCategory.TOKEN_CLASSIFICATION:
            dataset, label_map = self.remove_bio_encoding(dataset, label_column, label_map)

        self.log_dataset_info(
            text_column, label_column, label_map, task_category,
            self.dataset_downsample, dataset_size=len(dataset)
        )

        dataset = PreprocessedDataset(
            dataset=dataset,
            text_column=text_column,
            label_column=label_column,
            task_category=task_category,
        )

        return dataset

    @staticmethod
    def merge_dataset_splits(dataset: Union[str, Dataset, DatasetDict]) -> Dataset:
        if isinstance(dataset, DatasetDict):
            dataset = datasets.concatenate_datasets(list(dataset.values()))
        return dataset

    @staticmethod
    def find_column(column_role: str, dataset: Dataset) -> str:
        """Find text and label columns using common keywords."""
        common_names: dict = {
            'Text column': [
                "text", "sentence", "token", "tweet", "document", "paragraph", "description",
                "comment", "utterance", "question", "story", "context", "passage",
            ],
            "Label column": [
                "label", "ner_tag", "named_entities", "entities", "tag", "target", "category",
                "class", "sentiment", "polarity", "emotion", "rating", "stance",
            ]
        }

        columns = dataset.column_names
        found_column = next(
            (col for keyword in common_names[column_role] for col in columns if keyword in col),
            None
        )
        if found_column is None:
            raise ValueError(
                f"{column_role} not found in dataset: {dataset.column_names}. "
                f"Specify it manually text_column: str = ..."
            )

        return found_column

    @staticmethod
    def merge_text_pairs(text_column: str, text_pair_column: str, dataset: Dataset) -> Dataset:
        """Concatenate text pairs into a single text using separator token"""
        if text_pair_column not in dataset.column_names:
            raise ValueError(
                f"Text pair column name '{text_pair_column}' can not be found in the dataset. "
                f"Use one of the following names for tex pair: {dataset.column_names}."
            )

        def merge_texts(dataset_entry: dict[str, str]) -> dict[str, str]:
            dataset_entry[text_column] = (
                dataset_entry[text_column] + " [SEP] " + dataset_entry[text_pair_column]
            )
            return dataset_entry

        dataset = dataset.map(merge_texts, num_proc=None, desc="Merging text pair columns")
        new_text_column_name = text_column + "+" + text_pair_column
        dataset = dataset.rename_column(text_column, new_text_column_name)
        return dataset

    @staticmethod
    def find_task_category(label_column: str, dataset: Dataset) -> TaskCategory:
        """Determine task category based on the label column's data type."""
        label_to_task_type = {
            int: TaskCategory.TEXT_CLASSIFICATION,  # text classification labels can be integers
            str: TaskCategory.TEXT_CLASSIFICATION,  # or strings e.g. "positive"
            list: TaskCategory.TOKEN_CLASSIFICATION,  # token-level tasks have a list of labels
            float: TaskCategory.TEXT_REGRESSION,  # regression tasks have floats
        }

        label_type = type(dataset[label_column][0])

        for key, task_type in label_to_task_type.items():
            if issubclass(label_type, key):
                return task_type

        raise ValueError(
            f"Cannot determine task category for the label column '{label_column}'. "
            f"Label types are {list(label_to_task_type.keys())}, but got {label_type}."
        )

    @staticmethod
    def remove_empty_rows(
            text_column: str, label_column: str, dataset: Dataset
    ) -> Dataset:
        """Filter out entries with empty or noisy texts/labels."""
        def is_valid_entry(sample) -> bool:
            text, label = sample[text_column], sample[label_column]

            # Remove empty entries
            if not text or label is None:
                return False

            if not isinstance(text, list):
                text = [text]

            # Remove sentences with characters unsupported by most tokenizers
            _BAD_CHARACTERS = "\uFE0F"  # emoji variation symbol '\uFE0F', etc.
            if any(c in t for t in text for c in _BAD_CHARACTERS):
                return False

            if not isinstance(label, list):
                label = [label]

            # Remove negative labels from classification datasets
            if any(isinstance(word_label, int) and word_label < 0 for word_label in label):
                return False

            return True

        dataset = dataset.filter(is_valid_entry, desc="Removing empty rows")
        dataset = dataset.flatten_indices()
        return dataset

    @staticmethod
    def make_labels_categorical(label_column, label_map, dataset) -> Dataset:
        """Converts string labels to integers using a label map"""
        def convert_label(label):
            """Convert a label (string or list of strings) to its integer representation."""
            if isinstance(label, list):
                return [label_map[word_label] for word_label in label]
            return label_map[label]

        dataset = dataset.map(
            lambda x: {"label": convert_label(x[label_column])},
            desc="Converting labels to categorical"
        )

        return dataset

    @staticmethod
    def create_label_map(label_column: str, dataset: Dataset) -> dict[str, int]:
        """Find feature names to create a label map in a hf dataset."""
        label_names = getattr(
            getattr(dataset.features[label_column], "feature", None), "names", None
        ) or getattr(dataset.features[label_column], "names", None)

        # If label names are missing, create them manually
        if not label_names:
            label_names = sorted(
                {
                    str(label)
                    for sublist in dataset[label_column]
                    for label in (sublist if isinstance(sublist, list) else [sublist])
                }
            )

        label2id = {label: idx for idx, label in enumerate(label_names)}
        return label2id

    @staticmethod
    def downsample(ratio: float, dataset: Dataset) -> Dataset:
        """Reduce the dataset to a chosen ratio."""
        dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * ratio)))
        return dataset.flatten_indices()

    @staticmethod
    def remove_bio_encoding(
        dataset: Dataset, label_column: str, label_map: dict[str, int]
    ) -> tuple[Dataset, dict[str, int]]:
        """Remove BIO prefixes for NER labels and create a new label map."""
        unique_labels = set(label.split("-")[-1] for label in label_map)
        new_label_map = {label: idx for idx, label in enumerate(unique_labels)}

        # Map old ids to new ids
        reverse_map = {
            old_idx: new_label_map[label.split("-")[-1]] for label, old_idx in label_map.items()
        }
        dataset = dataset.map(
            lambda sample: {
                label_column: [reverse_map[old_idx] for old_idx in sample[label_column]]
            },
            desc="Removing BIO encoding",
        )

        # Check if label map was changed
        if label_map == new_label_map:
            logger.warning(
                "Could not remove BIO prefixes for this tagging dataset. Please add the correct "
                "label map as parameter label_map: dict[str, int] = ... manually."
            )

        return dataset, new_label_map

    @staticmethod
    def whitespace_tokenize(text_column: str, dataset: Dataset) -> Dataset:
        """Tokenize using Whitespace"""
        tokenizer = Whitespace()

        def pre_tokenize(example):
            encoding = tokenizer.pre_tokenize_str(example[text_column])
            example[text_column] = [token for token, _ in encoding]
            return example

        dataset = dataset.map(pre_tokenize, num_proc=None, desc="Whitespace pre-tokenization")
        return dataset

    @staticmethod
    def log_dataset_info(
            text_column, label_column, label_map, task_category, downsample_ratio, dataset_size
    ) -> None:
        """Log information about preprocessed dataset"""
        # Basic dataset configuration
        logger.info(
            f"Dataset Info - Text Column: {text_column}, Label Column: {label_column}, "
            f"Task Category: {task_category}, Dataset Size: {dataset_size} texts"
        )

        # Show the down-sampled size
        if downsample_ratio and downsample_ratio < 1.0:
            logger.info(
                f"Dataset has been downsampled to {int(downsample_ratio * 100)}% of original size."
            )

        # Log the label map
        if task_category != TaskCategory.TEXT_REGRESSION:
            logger.info(f"Label Map: {label_map}")
