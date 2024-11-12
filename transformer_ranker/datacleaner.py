import logging
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import datasets
import torch
from datasets.dataset_dict import Dataset, DatasetDict
from tokenizers.pre_tokenizers import Whitespace

from .utils import configure_logger

logger = configure_logger('transformer_ranker', logging.INFO)


class DatasetCleaner:
    def __init__(
        self,
        pre_tokenizer: Optional[Whitespace] = None,
        merge_data_splits: bool = True,
        remove_empty_sentences: bool = True,
        change_bio_encoding: bool = True,
        dataset_downsample: Optional[float] = None,
        task_type: Optional[str] = None,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        label_map: Optional[Dict[str, int]] = None,
        text_pair_column: Optional[str] = None,
    ):
        """
        Prepare huggingface dataset. Identify task type, find text and label columns, down-sample, merge data splits.

        :param pre_tokenizer: Pre-tokenizer to use, such as Whitespace from huggingface pre-tokenizers.
        :param merge_data_splits: Whether to merge train, dev, and test splits into one.
        :param change_bio_encoding: Change BIO encoding to single class labels by removing B-, I-, O- prefixes
        :param remove_empty_sentences: Whether to remove empty sentences.
        :param dataset_downsample: Fraction to reduce the dataset size.
        :param task_type: Task category (e.g., 'token classification', 'text classification', 'text regression').
        :param text_column: Column name for texts.
        :param label_column: Column name for labels.
        :param label_map: A dictionary which maps label names to integers.
        :param text_pair_column: Column name where the second text pair is stored (for entailment-like tasks)
        """
        self.pre_tokenizer = pre_tokenizer
        self.merge_data_splits = merge_data_splits
        self.change_bio_encoding = change_bio_encoding
        self.remove_empty_sentences = remove_empty_sentences
        self.dataset_downsample = dataset_downsample
        self.task_type = task_type
        self.text_column = text_column
        self.label_column = label_column
        self.label_map = label_map
        self.text_pair_column = text_pair_column
        self.dataset_size = 0

    def prepare_dataset(self, dataset: Union[str, DatasetDict, Dataset]) -> Union[Dataset, DatasetDict]:
        """Preprocess a dataset, leave only needed columns, down-sample

        :param dataset: dataset from huggingface. It can be one of the following:
        a DatasetDict (containing multiple splits) or a single dataset split (e.g., Dataset)
        :return: Return clean and preprocessed dataset, that can be later used in the transformer-ranker
        """
        # Load huggingface dataset
        dataset = datasets.load_dataset(dataset, trust_remote_code=True) if isinstance(dataset, str) else dataset

        # Ensure the dataset is either a DatasetDict (multiple splits) or a Dataset (single split)
        if not isinstance(dataset, (DatasetDict, Dataset)):
            raise ValueError(
                "The dataset must be an instance of either DatasetDict (for multiple splits) "
                "or Dataset (for a single split) to be preprocessed."
            )

        # Clone the dataset to avoid changing the original one
        dataset = dataset.map(lambda x: x, load_from_cache_file=False, desc="Cloning the dataset")

        if self.merge_data_splits and isinstance(dataset, DatasetDict):
            dataset = self._merge_data_splits(dataset)

        # Find text and label columns
        text_column, label_column, label_type = self._find_text_and_label_columns(dataset,
                                                                                  self.text_column,
                                                                                  self.label_column)

        # Find task category based on label type
        task_type = self._find_task_type(label_column, label_type) if not self.task_type else self.task_type

        # Clean the dataset by removing empty sentences and empty/negative labels
        if self.remove_empty_sentences:
            dataset = self._remove_empty_rows(dataset, text_column, label_column)

        # Down-sample the original dataset
        if self.dataset_downsample:
            dataset = self._downsample(dataset, ratio=self.dataset_downsample)

        # Pre-tokenize sentences if pre-tokenizer is given
        if not task_type == "token classification" and self.pre_tokenizer:
            dataset = self._tokenize(dataset, self.pre_tokenizer, text_column)

        # Concatenate text columns for text-pair tasks
        if self.text_pair_column:
            dataset, text_column = self._merge_textpairs(dataset, text_column, self.text_pair_column)

        # Convert string labels to integers
        if label_type == str:
            dataset, self.label_map = self._make_labels_categorical(dataset, label_column)

        # Try to find label map in the dataset
        if not self.label_map:
            self.label_map = self._create_label_map(dataset, label_column)

        # Remove BIO prefixes for ner or chunking tasks
        if task_type == "token classification" and self.change_bio_encoding:
            dataset, self.label_map = self._change_bio_encoding(dataset, label_column, self.label_map)

        logger.info(f"Label map: {self.label_map}")

        # Set updated attributes and log them
        self.text_column = text_column
        self.label_column = label_column
        self.task_type = task_type
        self.dataset_size = len(dataset)
        self.log_dataset_info()

        # Keep only text and label columns
        keep_columns = {self.text_column, self.text_pair_column, self.label_column} - {None}
        columns_to_remove = list(set(dataset.column_names) - keep_columns)
        dataset = dataset.remove_columns(columns_to_remove)

        return dataset

    def prepare_labels(self, dataset: Dataset) -> torch.Tensor:
        """Prepare labels as tensors.
        Flatten labels if they contain lists (for token classification)"""
        labels = dataset[self.label_column]
        labels = [item for sublist in labels for item in sublist] if isinstance(labels[0], list) else labels
        return torch.tensor(labels)

    def prepare_sentences(self, dataset: Dataset) -> List[str]:
        """Gather sentences in the text column."""
        return dataset[self.text_column]

    @staticmethod
    def _downsample(dataset: Dataset, ratio: float) -> Dataset:
        """Reduce the dataset to a chosen ratio."""
        dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * ratio)))
        return dataset

    @staticmethod
    def _find_text_and_label_columns(dataset: Dataset, text_column: Optional[str] = None,
                                     label_column: Optional[str] = None) -> Tuple[str, str, Type]:
        """Find text and label columns in hf datasets based on common keywords"""
        text_columns = [
            'text', 'sentence', 'token', 'tweet', 'document', 'paragraph', 'description', 'comment',
            'utterance', 'question', 'story', 'context', 'passage',
        ]

        label_columns = [
            'label', 'ner_tag', 'named_entities', 'entities', 'tag', 'target', 'category', 'class',
            'sentiment', 'polarity', 'emotion', 'rating', 'stance'
        ]

        column_names = dataset.column_names
        if not text_column:
            # Iterate over keywords and check if it exists in the dataset
            text_column = next((col for keyword in text_columns for col in column_names if keyword in col), None)
        if not label_column:
            label_column = next((col for keyword in label_columns for col in column_names if keyword in col), None)

        if not text_column or not label_column:
            missing = 'text' if not text_column else 'label'
            raise KeyError(f"Can not determine the {missing} column. Specify {missing}_column=\"...\" "
                           f"from available columns: {column_names}.")

        label_type = type(dataset[label_column][0])
        return text_column, label_column, label_type

    @staticmethod
    def _merge_textpairs(dataset: Dataset, text_column: str, text_pair_column: str) -> Tuple[Dataset, str]:
        """Concatenate text pairs into a single text using separator token"""
        new_text_column_name = text_column + '+' + text_pair_column

        def merge_texts(dataset_entry: Dict[str, str]) -> Dict[str, str]:
            dataset_entry[text_column] = dataset_entry[text_column] + " [SEP] " + dataset_entry[text_pair_column]
            dataset_entry[new_text_column_name] = dataset_entry.pop(text_column)
            return dataset_entry
        dataset = dataset.map(merge_texts, num_proc=None, desc="Merging text pair columns")
        return dataset, new_text_column_name

    @staticmethod
    def _find_task_type(label_column: str, label_type: type) -> str:
        """Determine the task type based on the label column's data type."""
        label_type_to_task_type = {
            int: "text classification",  # text classification labels can be integers
            str: "text classification",  # or strings e.g. "positive"
            list: "token classification",  # token-level tasks have a list of labels
            float: "text regression",  # regression tasks have continuous values
        }

        for key, task_type in label_type_to_task_type.items():
            if issubclass(label_type, key):
                return task_type

        raise ValueError(
            f"Cannot determine the task type for the label column '{label_column}'. "
            f"Expected label types are {list(label_type_to_task_type.keys())}, but got {label_type}."
        )

    @staticmethod
    def _tokenize(dataset: Dataset, pre_tokenizer: Whitespace, text_column: str) -> Dataset:
        """Tokenize a dataset using hf pre-tokenizer (e.g. Whitespace)"""
        def pre_tokenize(example):
            encoding = pre_tokenizer.pre_tokenize_str(example[text_column])
            example[text_column] = [token for token, _ in encoding]
            return example

        dataset = dataset.map(pre_tokenize, num_proc=None, desc="Pre-tokenizing texts using Whitespace")
        return dataset

    @staticmethod
    def _merge_data_splits(dataset: DatasetDict) -> Dataset:
        """Merge DatasetDict into a single dataset."""
        return datasets.concatenate_datasets(list(dataset.values()))

    @staticmethod
    def _remove_empty_rows(dataset: Dataset, text_column: str, label_column: str) -> Dataset:
        """Filter out entries with empty or invalid text or labels."""
        def is_valid_entry(sample) -> bool:
            text, label = sample[text_column], sample[label_column]

            # Check if text is non-empty and does not contain emoji variation char '\uFE0F'
            has_text = bool(text) and (not isinstance(text, list) or '\uFE0F' not in text)

            # Check if label is non-null and all elements are non-negative
            valid_label = label is not None and (not isinstance(label, list) or all(value >= 0 for value in label))

            return has_text and valid_label

        return dataset.filter(is_valid_entry, desc="Removing empty or invalid entries")

    @staticmethod
    def _make_labels_categorical(dataset: Dataset, label_column: str) -> Tuple[Dataset, Dict[str, int]]:
        """Convert string labels in the dataset to categorical integer labels, for classification tasks."""
        unique_labels = sorted(set(dataset[label_column]))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

        def map_labels(dataset_entry):
            dataset_entry[label_column] = label_map[dataset_entry[label_column]]
            return dataset_entry

        dataset = dataset.map(map_labels, num_proc=None, desc="Mapping string labels to integers")
        return dataset, label_map

    @staticmethod
    def _create_label_map(dataset: Dataset, label_column: str) -> Dict[str, int]:
        """Try to find feature names in a hf dataset"""
        label_names = (
                getattr(getattr(dataset.features[label_column], 'feature', None), 'names', None)
                or getattr(dataset.features[label_column], 'names', None)
                or sorted({str(label) for labels in dataset[label_column] for label in labels})
        )

        return {label: idx for idx, label in enumerate(label_names)}

    @staticmethod
    def _change_bio_encoding(
            dataset: Dataset,
            label_column: str,
            label_map: Dict[str, int]
    ) -> Tuple[Dataset, Dict[str, int]]:
        """Remove BIO prefixes from NER labels, update the dataset, and create a new label map."""

        # Get unique labels without BIO prefixes and create new label map
        unique_labels = set(label.split('-')[-1] for label in label_map)
        new_label_map = {label: idx for idx, label in enumerate(unique_labels)}

        # Map old ids to new ids
        reverse_map = {old_idx: new_label_map[label.split('-')[-1]] for label, old_idx in label_map.items()}
        dataset = dataset.map(
            lambda sample: {label_column: [reverse_map[old_idx] for old_idx in sample[label_column]]},
            desc="Removing BIO prefixes"
        )

        # Check if label map was changed
        if label_map == new_label_map:
            logger.warning("Could not remove BIO prefixes for this tagging dataset. "
                           "Please add the label map as parameter label_map: Dict[str, int] = ... manually.")

        return dataset, new_label_map

    def log_dataset_info(self) -> None:
        """Log information about dataset"""
        logger.info(f"Texts and labels: '{self.text_column}', '{self.label_column}'")
        logger.info(f"Task category: '{self.task_type}'")
        is_downsampled = self.dataset_downsample and self.dataset_downsample < 1.0
        downsample_info = f"(down-sampled to {self.dataset_downsample})" if is_downsampled else ""
        logger.info(f"Dataset size: {self.dataset_size} texts {downsample_info}")
