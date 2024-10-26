import torch
import datasets
from datasets.dataset_dict import DatasetDict, Dataset
from tokenizers.pre_tokenizers import Whitespace
from .utils import configure_logger

import logging
from typing import List, Dict, Optional, Set, Union, Tuple, Type


logger = configure_logger('transformer_ranker', logging.INFO)


class DatasetCleaner:
    def __init__(
        self,
        pre_tokenizer: Optional[Whitespace] = None,
        exclude_test_split: bool = False,
        merge_data_splits: bool = True,
        change_ner_encoding_to_spans: bool = True,
        remove_empty_sentences: bool = True,
        dataset_downsample: Optional[float] = None,
        task_type: Optional[str] = None,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        label_map: Optional[Dict[str, int]] = None,
        text_pair_column: Optional[str] = None,
    ):
        """
        Prepare huggingface dataset, clean it, find sentence and label columns.

        :param pre_tokenizer: Pre-tokenizer to use, such as Whitespace from huggingface pre-tokenizers.
        :param exclude_test_split: Whether to exclude the test split.
        :param merge_data_splits: Whether to merge train, dev, and test splits into one.
        :param change_ner_encoding_to_spans: Whether to change BIO encoding to single class labels.
        :param remove_empty_sentences: Whether to remove empty sentences.
        :param dataset_downsample: Fraction to downsample the dataset to.
        :param task_type: Type of task (e.g., 'sentence classification', 'word classification', 'sentence regression').
        :param text_column: Column name where texts are stored.
        :param label_column: Column name where labels are stored.
        :param label_map: Mapping of labels to integers.
        :param text_pair_column: Column name where the second text pair is stored. For entailment-type tasks.
        """
        self.pre_tokenizer = pre_tokenizer
        self.exclude_test_split = exclude_test_split
        self.merge_data_splits = merge_data_splits
        self.change_ner_encoding_to_spans = change_ner_encoding_to_spans
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

        if self.merge_data_splits and (isinstance(dataset, DatasetDict) or isinstance(dataset, list)):
            dataset = self._merge_data_splits(dataset)

        # Find text and label columns
        text_column, label_column, label_type = self._find_text_and_label_columns(dataset,
                                                                                  self.text_column,
                                                                                  self.label_column)

        # Determine task type based on label type if not specified
        task_type = self._find_task_type(label_column, label_type) if not self.task_type else self.task_type

        # Clean the dataset by removing empty sentences and negative labels
        if self.remove_empty_sentences:
            dataset = self._remove_empty_rows(dataset, text_column, label_column)

        # Down-sample the original dataset
        if self.dataset_downsample:
            dataset = self._downsample(dataset, ratio=self.dataset_downsample)

        # Pre-tokenize sentences if pre-tokenizer is specified
        if not task_type == "word classification" and self.pre_tokenizer:
            dataset = self._tokenize(dataset, self.pre_tokenizer, text_column)

        # Concatenate text columns for text-pair tasks
        if self.text_pair_column:
            dataset = self._merge_textpairs(dataset, text_column, self.text_pair_column)

        # Convert string labels to integers
        if label_type == str:
            dataset, label_map = self._make_labels_categorical(dataset, label_column)
            logger.info(f"Label map: {label_map}")

        # Change NER encoding to spans if specified
        if task_type == "word classification" and self.change_ner_encoding_to_spans:
            dataset, self.label_map = self._change_to_span_encoding(dataset, label_column, self.label_map)

        # Store updated attributes and log them
        self.text_column = text_column
        self.label_column = label_column
        self.task_type = task_type
        self.dataset_size = len(dataset)
        self.log_dataset_info(dataset)

        # Simplify the dataset: keep only relevant columns
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
    def _merge_textpairs(dataset: Dataset, text_column: str, text_pair_column: str) -> Dataset:
        """Concatenate text pairs into a single text using separator token"""
        def merge_texts(example: Dict[str, str]) -> Dict[str, str]:
            example[text_column] = example[text_column] + " [SEP] " + example[text_pair_column]
            return example
        dataset = dataset.map(merge_texts, num_proc=None, desc="Merge sentence pair columns")
        return dataset

    @staticmethod
    def _find_task_type(label_column: str, label_type: Union[Type[int], Type[str], Type[list], Type[float]]) -> str:
        """Determine task type based on the label column's data type."""
        label_type_to_task_type = {
            int: "sentence classification",  # labels can be integers
            str: "sentence classification",  # or strings e.g. "positive"
            list: "word classification",
            float: "sentence regression",
        }

        task_type = label_type_to_task_type.get(label_type, None)

        if not task_type:
            raise ValueError(f"Cannot determine task type from the label column '{label_column}' "
                             f"value: {type(label_type)}.")
        return task_type

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
            return entry_has_text and label_is_valid and all_tokens_are_valid # keep entries that have text and labels

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
    def _change_to_span_encoding(
        dataset: Dataset,
        label_column: str,
        label_map: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dataset, Dict[str, int]]:
        """Remove BIO prefixes for NER labels and create a new label map.
        Example: ['B-PER', 'I-PER', 'O'] -> ['PER', 'PER', 'O']
        Original label map: {'B-PER': 0, 'I-PER': 1, 'O': 2}
        Converted span label map: {'PER': 0, 'O': 1}

        :param dataset: The dataset containing BIO labels.
        :param label_column: The name of the label column.
        :param label_map: Optional dictionary to map BIO labels to integers. If not provided, a new one will be created.
        :return: A tuple with the dataset containing new labels and the updated label map.
        """
        # Attempt to get the label map from dataset features information
        if not label_map:
            features = dataset.features
            if label_column in features and hasattr(features[label_column], 'feature') and hasattr(
                    features[label_column].feature, 'names'):
                label_map = {name: idx for idx, name in enumerate(features[label_column].feature.names)}
            else:
                # Create label map manually if not found
                logger.info('Label map not found. Creating manually...')
                unique_labels: Set[str] = set()
                label_data = dataset[label_column] if isinstance(dataset, Dataset) else [dataset[split][label_column]
                                                                                         for split in dataset]
                for label_list in label_data:
                    unique_labels.update(
                        label.split('-')[-1] if isinstance(label, str) else str(label) for label in label_list)
                label_map = {label: idx for idx, label in enumerate(sorted(unique_labels, key=int))}

        logger.info(f"Label map: {label_map}")

        # Remove BIO encoding from the label map
        span_label_map: Dict[str, int] = {}
        for label in label_map:
            main_label = label.split('-')[-1] if isinstance(label, str) else label
            if main_label not in span_label_map:
                span_label_map[main_label] = len(span_label_map)

        logger.info(f"Simplified label map: {span_label_map}")

        if label_map == span_label_map:
            logger.warning("Could not convert BIO labels to span labels. "
                           "Please add the label map as parameter label_map: Dict[str, int] = ... manually.")

        # Create a reverse map from the original integer labels to the simplified span labels
        reverse_map = {}
        for original_label, index in label_map.items():
            main_label = original_label.split('-')[-1] if isinstance(original_label, str) else original_label
            reverse_map[index] = span_label_map[main_label]

        # Map labels to their corresponding span encoding
        def map_to_spans(example):
            example_labels = example[label_column]
            new_labels = [reverse_map[bio_label] for bio_label in example_labels]
            example[label_column] = new_labels
            return example

        if isinstance(dataset, DatasetDict):
            for split in dataset.keys():
                dataset[split] = dataset[split].map(map_to_spans, num_proc=None, desc="Mapping BIO to span encoding")
        else:
            dataset = dataset.map(map_to_spans, num_proc=None, desc="Mapping BIO to span encoding")

        return dataset, span_label_map

    def log_dataset_info(self, dataset) -> None:
        """Log information about dataset after cleaning it"""
        logger.info(f"Sentence and label columns: '{self.text_column}', '{self.label_column}'")
        logger.info(f"Task type: '{self.task_type}'")
        downsample_info = f"(downsampled to {self.dataset_downsample})" if self.dataset_downsample else ""
        logger.info(f"Dataset size: {self.dataset_size} {downsample_info}")
