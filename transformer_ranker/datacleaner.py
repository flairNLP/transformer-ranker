import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import datasets
import torch
from datasets.dataset_dict import Dataset, DatasetDict

from .utils import configure_logger

logger = configure_logger("transformer_ranker", logging.INFO)


class TaskCategory(str, Enum):
    """Supported task categories"""

    TEXT_CLASSIFICATION = "text classification"
    TEXT_PAIR_CLASSIFICATION = "text pair classification"
    TEXT_PAIR_REGRESSION = "text pair regression"
    TEXT_REGRESSION = "text regression"
    TOKEN_CLASSIFICATION = "token classification"

    def is_classification_task(self):
        return "classification" in self.value


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
        """Prepare texts and labels, assign task category.

        Downsample dataset, find text and label columns, create label map,
        preprocess labels, pre-tokenize, clean rows, merge text pair columns.
        Returns: (processed texts, label tensor, task category)
        """

        # Verify dataset type
        if not isinstance(dataset, (Dataset, DatasetDict)):
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

        # Merge splits into one
        if isinstance(dataset, DatasetDict):
            dataset = datasets.concatenate_datasets(list(dataset.values()))

        # Find text and label fields, set the task category
        text_column = self.text_column or self._find_column(dataset, "text column")
        label_column = self.label_column or self._find_column(dataset, "label column")
        task_category = self.task_category or self._find_task_category(dataset, label_column, self.text_pair_column)

        # Merge text pair columns using a separator
        if self.text_pair_column:
            dataset = self._merge_text_pairs(dataset, text_column, self.text_pair_column)
            text_column = f"{text_column}_with_{self.text_pair_column}"

        # Remove unused columns
        dataset = dataset.select_columns([text_column, label_column])

        # Downsample to a given ratio
        if self.dataset_downsample:
            dataset = self._downsample(dataset, self.dataset_downsample)

        # Clean noisy or empty rows
        if self.cleanup_rows:
            dataset = self._cleanup_rows(dataset, text_column, label_column)

        # Set or create label map for classification tasks
        label_map = self.label_map
        if task_category.is_classification_task():
            dataset, label_map = (dataset, label_map) if label_map else self._create_label_map(dataset, label_column)

            # Remove BIO encoding for token classification
            if task_category == TaskCategory.TOKEN_CLASSIFICATION and self.remove_bio_encoding:
                dataset, label_map = self._remove_bio_encoding(dataset, label_column, label_map)

        # Prepare all texts and labels as tensors
        texts = dataset[text_column]
        labels = dataset[label_column]
        if task_category == TaskCategory.TOKEN_CLASSIFICATION:
            labels = [word_label for labels in dataset[label_column] for word_label in labels]
        labels = torch.tensor(labels)

        # Log dataset info
        self._log_dataset_info(
            text_column,
            label_column,
            label_map,
            task_category,
            self.dataset_downsample,
            dataset_size=len(dataset),
        )

        return texts, labels, task_category

    @staticmethod
    def _find_column(dataset: Dataset, column_role: str) -> str:
        """Find text and label columns using common keywords."""
        common_names: dict = {
            'text column': [
                "text", "sentence", "token", "tweet", "document", "paragraph", "description",
                "comment", "utterance", "question", "story", "context", "passage",
            ],
            "label column": [
                "label", "ner_tag", "named_entities", "entities", "tag", "target", "category",
                "class", "sentiment", "polarity", "emotion", "rating", "stance",
            ]
        }  # fmt: skip

        columns = dataset.column_names
        found_column = next((col for keyword in common_names[column_role] for col in columns if keyword in col), None)
        if found_column is None:
            raise ValueError(
                f"{column_role} not found in dataset: {dataset.column_names}. "
                f"Set the text_column: str = ... manually."
            )

        return found_column

    @staticmethod
    def _merge_text_pairs(dataset: Dataset, text_column: str, text_pair_column: str) -> Dataset:
        """Concatenate text pairs into single column using sep token"""
        if text_pair_column not in dataset.column_names:
            raise ValueError(
                f"Text pair column name '{text_pair_column}' can not be found in the dataset. "
                f"Use one of the following names for tex pair: {dataset.column_names}."
            )

        dataset = dataset.map(
            lambda row: {text_column: row[text_column] + " [SEP] " + row[text_pair_column]},
            desc="Merging text pair columns",
        )

        new_text_column_name = text_column + "_with_" + text_pair_column
        dataset = dataset.rename_column(text_column, new_text_column_name)
        return dataset

    @staticmethod
    def _find_task_category(
        dataset: Dataset, label_column: str, text_pair_column: Optional[str] = None
    ) -> TaskCategory:
        """Assign task category based on label type."""
        label_to_task_category = {
            int: TaskCategory.TEXT_CLASSIFICATION,
            str: TaskCategory.TEXT_CLASSIFICATION,
            list: TaskCategory.TOKEN_CLASSIFICATION,
            float: TaskCategory.TEXT_REGRESSION,
        }

        # Label type should be consistent across dataset
        label_types = list(set(type(label) for label in dataset[label_column]))
        label_type = label_types[0]
        if len(label_types) != 1:
            raise ValueError(f"Inconsistent label types: {label_types}. Ensure all labels are the same.")

        task_category = label_to_task_category.get(label_type, None)
        if not task_category:
            raise ValueError(
                f"Can't determine task category for {label_type}. Supported: {list(label_to_task_category.keys())}."
            )

        # If text pairs are present, adjust category
        if text_pair_column in dataset.column_names:
            pair_mapping = {
                TaskCategory.TEXT_CLASSIFICATION: TaskCategory.TEXT_PAIR_CLASSIFICATION,
                TaskCategory.TEXT_REGRESSION: TaskCategory.TEXT_PAIR_REGRESSION,
            }
            task_category = pair_mapping.get(task_category, task_category)

        return task_category

    @staticmethod
    def _downsample(dataset: Dataset, ratio: float) -> Dataset:
        """Reduce dataset size to given ratio."""
        return dataset.shuffle(seed=42).select(range(int(len(dataset) * ratio)))

    @staticmethod
    def _cleanup_rows(dataset: Dataset, text_column: str, label_column: str) -> Dataset:
        """Filter out entries with empty or noisy texts and labels."""

        def is_valid_entry(dataset_row) -> bool:
            text, label = dataset_row[text_column], dataset_row[label_column]

            if not text or label is None:
                return False

            if not isinstance(text, list):
                text = [text]

            bad_characters = ["\ufe0f"]  # emoji variation symbol '\uFE0F'
            if any(char in t for t in text for char in bad_characters):
                return False

            if not isinstance(label, list):
                label = [label]

            # Remove "-1" labels sometimes used for unlabeled text
            if any(word_label == -1 for word_label in label):
                return False

            return True

        dataset = dataset.filter(is_valid_entry, desc="Removing empty rows")
        return dataset

    @staticmethod
    def _map_string_labels_to_integers(dataset, label_column) -> tuple[Dataset, dict[str, int]]:
        """Convert string labels to integers and retain label map."""
        label_names = sorted(set(dataset[label_column]))
        label_map = {label_name: idx for idx, label_name in enumerate(label_names)}

        def label_to_id(label):
            if isinstance(label, list):
                return [label_map[word_label] for word_label in label]
            return label_map[label]

        dataset = dataset.map(
            lambda dataset_row: {label_column: label_to_id(dataset_row[label_column])},
            desc="Converting string labels to integers",
        )

        return dataset, label_map

    @staticmethod
    def _create_label_map(dataset: Dataset, label_column: str) -> tuple[Dataset, dict[str, int]]:
        """Find feature names to create label map, convert labels to integers if needed."""
        label_names = getattr(getattr(dataset.features[label_column], "feature", None), "names", None) or getattr(
            dataset.features[label_column], "names", None
        )

        if not label_names:
            label_names = sorted(
                set(
                    label
                    for sublist in dataset[label_column]
                    for label in (sublist if isinstance(sublist, list) else [sublist])
                )
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
                desc="Converting string labels to integers",
            )
        return dataset, label_map

    @staticmethod
    def _remove_bio_encoding(
        dataset: Dataset, label_column: str, label_map: dict[str, int]
    ) -> tuple[Dataset, dict[str, int]]:
        """Remove BIO prefixes for ner labels and create new label map."""
        labels = [label.split("-")[-1] for label in label_map.keys()]
        unique_labels = list(dict.fromkeys(labels))
        new_label_map = {label: idx for idx, label in enumerate(unique_labels)}

        if label_map == new_label_map:
            logger.warning(
                "Could not remove BIO encoding. Pass your own label map "
                "when initializing the ranker label_map: dict[str, int] = ..."
            )

        new_idx = {old_idx: new_label_map[label.split("-")[-1]] for label, old_idx in label_map.items()}

        dataset = dataset.map(
            lambda dataset_entry: {label_column: [new_idx[old_idx] for old_idx in dataset_entry[label_column]]},
            desc="Removing BIO encoding",
        )

        return dataset, new_label_map

    @staticmethod
    def _log_dataset_info(text_column, label_column, label_map, task_category, downsample_ratio, dataset_size) -> None:
        """Log information about preprocessed dataset"""
        logger.info(f"Task category: {task_category.value}")
        logger.info(f"Text column: '{text_column}', label column: '{label_column}'")

        if downsample_ratio and downsample_ratio < 1.0:
            percentage = downsample_ratio * 100
            logger.info(f"Dataset size: {dataset_size} texts, reduced to {percentage:1g}% of original.")

        if task_category.is_classification_task():
            logger.info(f"Label map: {label_map}")
