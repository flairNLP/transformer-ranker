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
        exclude_test_split: bool = False,
        merge_data_splits: bool = True,
        remove_empty_sentences: bool = True,
        dataset_downsample: Optional[float] = None,
        task_type: Optional[str] = None,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        label_map: Optional[Dict[str, int]] = None,
        text_pair_column: Optional[str] = None,
        remove_bio_notation: bool = True,
    ):
        """
        Prepare huggingface dataset. Identify task type, find text and label columns, down-sample, merge data splits.

        :param pre_tokenizer: Pre-tokenizer to use, such as Whitespace from huggingface pre-tokenizers.
        :param exclude_test_split: Whether to exclude the test split.
        :param merge_data_splits: Whether to merge train, dev, and test splits into one.
        :param remove_bio_notation: Change BIO encoding to single class labels by removing B-, I-, O- prefixes
        :param remove_empty_sentences: Whether to remove empty sentences.
        :param dataset_downsample: Fraction to reduce the dataset size.
        :param task_type: Task category (e.g., 'token classification', 'text classification', 'text regression').
        :param text_column: Column name for texts.
        :param label_column: Column name for labels.
        :param label_map: A dictionary which maps label names to integers.
        :param text_pair_column: Column name where the second text pair is stored (for entailment-like tasks)
        """
        self.pre_tokenizer = pre_tokenizer
        self.exclude_test_split = exclude_test_split
        self.merge_data_splits = merge_data_splits
        self.remove_bio_notation = remove_bio_notation
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

        # Remove test split if specified
        if self.exclude_test_split and 'test' in dataset:
            logger.info("Removing the test split")
            del dataset['test']

        if self.merge_data_splits and isinstance(dataset, DatasetDict):
            dataset = self._merge_data_splits(dataset)

        # Find text and label columns
        text_column, label_column, label_type = self._find_text_and_label_columns(dataset,
                                                                                  self.text_column,
                                                                                  self.label_column)

        # Find task type based on label type
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
            dataset, label_map = self._make_labels_categorical(dataset, label_column)
            logger.info(f"Label map: {label_map}")

        # Remove BIO prefixes for ner or chunking tasks
        if task_type == "token classification" and self.remove_bio_notation:
            dataset, self.label_map = self._remove_bio_notation(dataset, label_column, self.label_map)

        # Set updated attributes and log them
        self.text_column = text_column
        self.label_column = label_column
        self.task_type = task_type
        self.dataset_size = len(dataset)
        self.log_dataset_info()

        # Keep only text and label columns
        keep_columns = [col for col in (self.text_column, self.text_pair_column, self.label_column) if col is not None]
        dataset = self._remove_columns(dataset, keep_columns=keep_columns)

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
        """Determine text and label columns in hf datasets based on popular keywords"""
        # A list of mostly used column names for texts
        text_columns = [
            'text', 'sentence', 'token', 'tweet', 'document', 'paragraph', 'description', 'comment',
            'utterance', 'question', 'story', 'context', 'passage',
        ]

        # A list of mostly used column names for labels
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

        def merge_texts(example: Dict[str, str]) -> Dict[str, str]:
            example[text_column] = example[text_column] + " [SEP] " + example[text_pair_column]
            example[new_text_column_name] = example.pop(text_column)
            return example
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
        """Remove entries with empty sentences or labels."""
        def dataset_row_is_clean(example) -> bool:
            text = example[text_column]
            label = example[label_column]
            entry_has_text = bool(text) if isinstance(text, list) else True  # non empty string
            all_tokens_are_valid = all(token != '\uFE0F' for token in text) if isinstance(text, list) else True
            label_is_valid = label is not None and (all(l >= 0 for l in label) if isinstance(label, list) else label >= 0)
            return entry_has_text and label_is_valid and all_tokens_are_valid  # keep entries that have text and labels

        dataset = dataset.filter(dataset_row_is_clean, desc="Removing empty sentences")
        return dataset

    @staticmethod
    def _remove_columns(dataset: Dataset, keep_columns: List[str]) -> Dataset:
        """Remove columns from the dataset that are not in the keep_columns list
        (e.g. remove all columns except the text and the label column)"""
        columns_to_remove = [col for col in dataset.column_names if col not in keep_columns]
        dataset = dataset.remove_columns(columns_to_remove)
        return dataset

    @staticmethod
    def _make_labels_categorical(dataset: Dataset, label_column: str) -> Tuple[Dataset, Dict[str, int]]:
        """Convert string labels in the dataset to categorical integer labels, for classification tasks.

        :param dataset: The dataset containing the labels to convert.
        :param label_column: The column name in the dataset that contains the labels.
        :return: A tuple with the new dataset (with integer labels) and the label map.
        """
        unique_labels = sorted(set(dataset[label_column]))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

        def map_labels(example):
            example[label_column] = label_map[example[label_column]]
            return example

        dataset = dataset.map(map_labels, num_proc=None, desc="Mapping string labels to integers")
        return dataset, label_map

    @staticmethod
    def _remove_bio_notation(
        dataset: Dataset,
        label_column: str,
        label_map: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dataset, Dict[str, int]]:
        """Remove BIO prefixes for NER labels and create a new label map.
        Example: ['O', 'B-PER', 'I-PER'] -> ['O', 'PER', 'PER']
        Original label map: {'O': 0, 'B-PER': 1, 'I-PER': 2}
        Label map without BIO notation: {'O': 0, 'PER': 1}

        :param dataset: The dataset containing BIO labels.
        :param label_column: The name of the label column.
        :param label_map: Optional dictionary to map BIO labels to integers. If not provided, a new one will be created.
        :return: A tuple with the dataset containing new labels and the updated label map.
        """
        if not label_map:
            try:
                # Attempt to get the label map from dataset feature information
                label_map = {label: idx for idx, label in enumerate(dataset.features[label_column].feature.names)}
            except AttributeError:
                # Try to create label map manually
                logger.info('Label map not found. Creating manually...')
                unique_labels: Set[str] = set()

                for label_list in dataset[label_column]:
                    unique_labels.update(
                        label.split('-')[-1] if isinstance(label, str) else str(label) for label in label_list)
                label_map = {label: idx for idx, label in enumerate(sorted(unique_labels, key=int))}

        # Remove BIO encoding and create a new label map
        new_label_map: Dict[str, int] = {}
        for label in label_map:
            main_label = label.split('-')[-1] if isinstance(label, str) else label
            if main_label not in new_label_map:
                new_label_map[main_label] = len(new_label_map)

        # Create a reverse map from original integer labels to labels without BIO prefixes
        reverse_map = {}
        for original_label, index in label_map.items():
            main_label = original_label.split('-')[-1] if isinstance(original_label, str) else original_label
            reverse_map[index] = new_label_map[main_label]

        # Map labels to their class labels without BIO
        def map_to_spans(example):
            example_labels = example[label_column]
            new_labels = [reverse_map[bio_label] for bio_label in example_labels]
            example[label_column] = new_labels
            return example

        if isinstance(dataset, DatasetDict):
            for split in dataset.keys():
                dataset[split] = dataset[split].map(map_to_spans, num_proc=None, desc="Removing BIO encoding")
        else:
            dataset = dataset.map(map_to_spans, num_proc=None, desc="Removing BIO encoding")

        if label_map == new_label_map:
            logger.warning("Could not remove BIO prefixes for this tagging dataset. "
                           "Please add the label map as parameter label_map: Dict[str, int] = ... manually.")
        else:
            logger.info(f"Label map: {label_map}")
            logger.info(f"New label map: {new_label_map}")

        return dataset, new_label_map

    def log_dataset_info(self) -> None:
        """Log information about dataset"""
        logger.info(f"Texts and labels: '{self.text_column}', '{self.label_column}'")
        logger.info(f"Task category: '{self.task_type}'")
        is_downsampled = self.dataset_downsample and self.dataset_downsample < 1.0
        downsample_info = f"(down-sampled to {self.dataset_downsample})" if is_downsampled else ""
        logger.info(f"Dataset size: {self.dataset_size} texts {downsample_info}")
