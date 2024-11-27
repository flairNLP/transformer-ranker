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
    """Supported task categories"""
    TOKEN_CLASSIFICATION = "token classification"
    TEXT_CLASSIFICATION = "text classification"
    TEXT_REGRESSION = "text regression"

    def __str__(self):
        return self.value


@dataclass
class DatasetCleaner:
    dataset_downsample: Optional[float] = None
    text_column: Optional[str] = None
    text_pair_column: Optional[str] = None
    label_column: Optional[str] = None
    label_map: Optional[dict] = None
    task_category: Optional[TaskCategory] = None
    cleanup_rows: bool = True
    remove_bio_encoding: bool = True
    tokenize: bool = False

    def prepare_dataset(
        self, dataset: Union[str, Dataset, DatasetDict]
    ) -> tuple[Union[list[str], list[list[str]]], torch.Tensor, TaskCategory]:
        """Prepare texts and labels, and assign a task category.

        Downsample the dataset, find text and label columns, create label map, 
        preprocess labels, pre-tokenize, clean rows, merge columns.
        Returns: (processed texts, label tensor, task category)
        """

        # Verify dataset type
        if not isinstance(dataset, (Dataset, DatasetDict)):
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

        # Merge dataset splits into one
        if isinstance(dataset, DatasetDict):
            dataset = datasets.concatenate_datasets(list(dataset.values()))

        # Find or set the text field
        text_column = self.text_column if self.text_column \
            else self._find_column("Text column", dataset)

        # Find or set the label field
        label_column = self.label_column if self.label_column \
            else self._find_column("Label column", dataset)

        # Find or set the task_category
        task_category = self.task_category if self.task_category \
            else self._find_task_category(label_column, dataset)

        # Set or create a label map for classification
        label_map, dataset = (self.label_map, dataset) if self.label_map \
            else self._create_label_map(label_column, dataset)

        # Combine text pair columns with a separator token
        if self.text_pair_column:
            dataset = self._merge_text_pairs(text_column, self.text_pair_column, dataset)
            text_column = f"{text_column}+{self.text_pair_column}"

        # Remove unused columns
        dataset = dataset.select_columns([text_column, label_column])

        # Downsample to a given ratio
        if self.dataset_downsample:
            dataset = self._downsample(self.dataset_downsample, dataset)

        # Clean noisy or empty rows
        if self.cleanup_rows:
            dataset = self._cleanup_rows(text_column, label_column, dataset)

        # Optional pre-tokenization
        if self.tokenize and isinstance(dataset[text_column][0], str):
            dataset = self._whitespace_tokenize(text_column, dataset)

        # Handle BIO encoding for token classification
        if self.remove_bio_encoding and task_category == TaskCategory.TOKEN_CLASSIFICATION:
            dataset, label_map = self._remove_bio_encoding(dataset, label_column, label_map)

        # Prepare all texts
        texts = dataset[text_column]

        # Prepare all labels as a tensor
        labels = dataset[label_column]
        if task_category == TaskCategory.TOKEN_CLASSIFICATION:
            labels = [word_label for labels in dataset[label_column] for word_label in labels]
        labels = torch.tensor(labels)

        # Log some preprocessed dataset info
        self._log_dataset_info(
            text_column, label_column, label_map, task_category,
            self.dataset_downsample, dataset_size=len(dataset)
        )

        return texts, labels, task_category

    @staticmethod
    def _find_column(column_role: str, dataset: Dataset) -> str:
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
    def _merge_text_pairs(text_column: str, text_pair_column: str, dataset: Dataset) -> Dataset:
        """Concatenate text pairs into a single string using separator token"""
        if text_pair_column not in dataset.column_names:
            raise ValueError(
                f"Text pair column name '{text_pair_column}' can not be found in the dataset. "
                f"Use one of the following names for tex pair: {dataset.column_names}."
            )

        dataset = dataset.map(
            lambda dataset_row: {
                text_column: dataset_row[text_column] + " [SEP] " + dataset_row[text_pair_column]
            },
            desc="Merging text pair columns",
        )

        new_text_column_name = text_column + "+" + text_pair_column
        dataset = dataset.rename_column(text_column, new_text_column_name)
        return dataset

    @staticmethod
    def _find_task_category(label_column: str, dataset: Dataset) -> TaskCategory:
        """Determine task category based on the label column's data type."""
        label_to_task_category = {
            int: TaskCategory.TEXT_CLASSIFICATION,  # text classification labels can be integers
            str: TaskCategory.TEXT_CLASSIFICATION,  # or strings e.g. "positive"
            list: TaskCategory.TOKEN_CLASSIFICATION,  # token-level tasks have a list of labels
            float: TaskCategory.TEXT_REGRESSION,  # regression tasks have floats
        }

        label_types = list(set(type(label) for label in dataset[label_column]))

        if len(label_types) != 1:
            raise ValueError(
                f"The dataset has inconsistent types in the label column: {label_types}. "
                f"All labels should have the same type."
            )

        for key, task_category in label_to_task_category.items():
            if issubclass(label_types[0], key):
                return task_category

        raise ValueError(
            f"Cannot determine task category for the label column '{label_column}'. "
            f"Supported label types for are {list(label_to_task_category.keys())}."
        )

    @staticmethod
    def _cleanup_rows(
            text_column: str, label_column: str, dataset: Dataset
    ) -> Dataset:
        """Filter out entries with empty or noisy texts and labels."""
        def is_valid_entry(dataset_row) -> bool:
            text, label = dataset_row[text_column], dataset_row[label_column]

            # Remove empty entries
            if not text or label is None:
                return False

            if not isinstance(text, list):
                text = [text]

            # Remove sentences with characters unsupported by most tokenizers
            bad_characters = ["\uFE0F"]  # emoji variation symbol '\uFE0F'
            if any(char in t for t in text for char in bad_characters):
                return False

            if not isinstance(label, list):
                label = [label]

            # Remove negative labels from classification datasets
            if any(isinstance(word_label, int) and word_label < 0 for word_label in label):
                return False

            return True

        dataset = dataset.filter(is_valid_entry, desc="Removing empty rows")
        return dataset

    @staticmethod
    def _map_string_labels_to_integers(label_column, dataset) -> tuple[Dataset, dict[str, int]]:
        """Converts string labels to integers and store the label map"""
        label_names = sorted(set(dataset[label_column]))
        label_map = {label_name: idx for idx, label_name in enumerate(label_names)}

        def label_to_id(label):
            if isinstance(label, list):
                return [label_map[word_label] for word_label in label]
            return label_map[label]

        dataset = dataset.map(
            lambda dataset_row: {label_column: label_to_id(dataset_row[label_column])},
            desc="Converting string labels to integers"
        )

        return dataset, label_map

    @staticmethod
    def _create_label_map(label_column: str, dataset: Dataset) -> tuple[dict[str, int], Dataset]:
        """Find feature names to create a label map in a hf dataset.
        Convert label column to integers if needed."""
        label_names = getattr(
            getattr(dataset.features[label_column], "feature", None), "names", None
        ) or getattr(dataset.features[label_column], "names", None)

        if not label_names:
            label_names = sorted(
                {
                    label for sublist in dataset[label_column]
                    for label in (sublist if isinstance(sublist, list) else [sublist])
                }
            )
            label_names = [str(label) for label in label_names]

        label_map = {label: idx for idx, label in enumerate(label_names)}

        if type(dataset[label_column][0]) in (str, list[str]):
            dataset = dataset.map(
                lambda dataset_row: {
                    label_column: (
                        label_map[dataset_row[label_column]]
                        if isinstance(dataset_row[label_column], str)
                        else [label_map[word_label] for word_label in label_column]
                    )
                },
                desc="Converting string labels to integers"
            )
        return label_map, dataset

    @staticmethod
    def _downsample(ratio: float, dataset: Dataset) -> Dataset:
        """Reduce the dataset to a chosen ratio."""
        return dataset.shuffle(seed=42).select(range(int(len(dataset) * ratio)))

    @staticmethod
    def _remove_bio_encoding(
        dataset: Dataset, label_column: str, label_map: dict[str, int]
    ) -> tuple[Dataset, dict[str, int]]:
        """Remove BIO prefixes for NER labels and create a new label map."""
        labels = [label.split("-")[-1] for label in label_map.keys()]
        unique_labels = list(dict.fromkeys(labels))
        new_label_map = {label: idx for idx, label in enumerate(unique_labels)}

        # Check if label map was changed
        if label_map == new_label_map:
            logger.warning(
                "Could not remove BIO encoding. Pass your own label map "
                "when initializing the ranker label_map: dict[str, int] = ..."
            )

        new_idx = {
            old_idx: new_label_map[label.split("-")[-1]] for label, old_idx in label_map.items()
        }

        dataset = dataset.map(
            lambda dataset_entry: {
                label_column: [new_idx[old_idx] for old_idx in dataset_entry[label_column]]
            },
            desc="Removing BIO encoding",
        )

        return dataset, new_label_map

    @staticmethod
    def _whitespace_tokenize(text_column: str, dataset: Dataset) -> Dataset:
        """Tokenize using Whitespace"""
        tokenizer = Whitespace()

        def pre_tokenize(example):
            encoding = tokenizer.pre_tokenize_str(example[text_column])
            example[text_column] = [token for token, _ in encoding]
            return example

        dataset = dataset.map(pre_tokenize, num_proc=None, desc="Whitespace pre-tokenization")
        return dataset

    @staticmethod
    def _log_dataset_info(
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
