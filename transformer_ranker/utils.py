import torch
import datasets
from datasets.dataset_dict import DatasetDict, Dataset
from tokenizers.pre_tokenizers import Whitespace

import logging
from typing import List, Dict, Optional, Union, Tuple, Any


def prepare_popular_models(model_size='base') -> List[str]:
    """Two lists of language models to try out"""
    base_models = [
        # English models
        "distilbert-base-cased",
        "typeform/distilroberta-base-v2",
        "bert-base-cased",
        "SpanBERT/spanbert-base-cased",
        "roberta-base",
        "google/electra-small-discriminator",
        "google/electra-base-discriminator",
        "microsoft/deberta-v3-base",
        # Sentence-transformers
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2",
        # Multilingual models
        "FacebookAI/xlm-roberta-base",
        "microsoft/mdeberta-v3-base",
        # German model
        "german-nlp-group/electra-base-german-uncased",
        # Domain-specific models
        "Lianglab/PharmBERT-cased",
        "Twitter/twhin-bert-base",
        "dmis-lab/biobert-base-cased-v1.2",
        "KISTI-AI/scideberta",
    ]

    large_models = [
        # English models
        "bert-large-uncased",
        "roberta-large",
        "google/electra-large-discriminator",
        "microsoft/deberta-v3-large",
        # Sentence transformers
        "sentence-transformers/all-mpnet-base-v2",
        # Multilingual models
        "FacebookAI/xlm-roberta-large",
        "microsoft/mdeberta-v3-base",
        # German model
        "deepset/gelectra-large",
        # Domain-specific models
        "dmis-lab/biobert-large-cased-v1.1",
        "Twitter/twhin-bert-large",
        "KISTI-AI/scideberta",
    ]

    return large_models if model_size == 'large' else base_models


class Result:
    def __init__(self, metric: str):
        """Store all rankings and transferability scores in Result.
        Includes scores for each layer in "layer_estimates".

        param metric: metric name (e.g. "hscore", or "logme")
        """
        self.metric = metric
        self._results = {}
        self.layer_estimates = {}

    @property
    def results(self) -> Dict[str, float]:
        """Return the result dictionary sorted by scores in descending order"""
        return dict(sorted(self._results.items(), key=lambda x: x[1], reverse=True))

    @property
    def best_model(self) -> str:
        """Return the model with the highest score"""
        model_name, _ = max(self.results.items(), key=lambda item: item[1])
        return model_name

    @property
    def top_three(self) -> Dict[str, float]:
        """Return first three model names and scores"""
        return {k: self.results[k] for k in list(self.results.keys())[:3]}

    @property
    def best_layers(self) -> Dict[str, str]:
        """Return a dictionary with model name: best layer id"""
        return {model: max(values, key=values.get) for model, values in self.layer_estimates.items()}

    def add_score(self, model_name, score) -> None:
        self._results[model_name] = score

    def append(self, additional_results: "Result") -> None:
        if isinstance(additional_results, Result):
            self._results.update(additional_results.results)
            self.layer_estimates.update(additional_results.layer_estimates)
        else:
            raise ValueError(f"Expected an instance of 'Result', but got {type(additional_results).__name__}. "
                             f"Only 'Result' instances can be appended.")

    def __str__(self) -> str:
        """Return sorted results as a string"""
        sorted_results = sorted(self._results.items(), key=lambda item: item[1], reverse=True)
        result_lines = [f"Rank {i+1}. {model_name}: {score}" for i, (model_name, score) in enumerate(sorted_results)]
        return "\n".join(result_lines)


def configure_logger(name: str, level: int = logging.INFO, log_to_console: bool = True) -> logging.Logger:
    """
    Configure transformer-ranker logger.

    :param name: The name of the logger.
    :param level: The logging level (default: logging.INFO).
    :param log_to_console: Whether to log to console (default: True)
    :return: Configured TransformerRanker logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            formatter = logging.Formatter('transformer_ranker:%(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    logger.propagate = False

    return logger


# Configure logger for the ranker
logger = configure_logger('transformer_ranker', logging.INFO)


class DatasetCleaner:
    def __init__(
        self,
        pre_tokenizer: Optional[Whitespace] = None,
        exclude_test_split: bool = False,
        merge_data_splits: bool = True,
        change_ner_encoding_to_spans: bool = True,
        remove_empty_sentences: bool = False,
        dataset_downsample: Optional[float] = None,
        task_type: Optional[str] = None,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        label_map: Optional[Dict[str, int]] = None,
        text_pair_column: Optional[str] = None,
    ):
        """
        We prepare any huggingface dataset for easy-to-use integration.

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

    def prepare_dataset(self, dataset: Union[str, DatasetDict]) -> Union[Dataset, DatasetDict]:
        """Preparing the dataset by data cleaning,

        :param dataset: The original dataset from huggingface
        :return: Return the cleaned dataset, that can be used by the ranker
        """
        # Load huggingface dataset
        dataset = datasets.load_dataset(dataset, trust_remote_code=True) if isinstance(dataset, str) else dataset

        # Clone the dataset to avoid modifying the original one
        dataset = dataset.map(lambda x: x, load_from_cache_file=False, desc="Cloning the dataset")

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

        # Remove test split if specified
        if self.exclude_test_split and 'test' in dataset:
            logger.info("Removing the test split")
            del dataset['test']

        # Pre-tokenize sentences if pre-tokenizer is specified
        if not task_type == "word classification" and self.pre_tokenizer:
            dataset = self._tokenize(dataset, self.pre_tokenizer, text_column)

        # Merge text pairs if text pair column is specified
        if self.text_pair_column:
            dataset = self._merge_textpairs(dataset, text_column, self.text_pair_column)

        # Merge data splits for estimation on all available data
        if self.merge_data_splits:
            dataset = self._merge_data_splits(dataset)

        # Convert string labels to integers if needed
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
        self.log_dataset_info()

        # Simplify the dataset: keep only relevant columns
        keep_columns = [self.text_column, self.text_pair_column, self.label_column]
        dataset = self._remove_columns(dataset, keep_columns=keep_columns)

        return dataset

    def prepare_labels(self, dataset: Union[DatasetDict, Dataset]) -> torch.Tensor:
        """Gather labels from the dataset and convert them to a tensor."""
        labels = []
        if isinstance(dataset, DatasetDict):
            for split in dataset.keys():
                labels.extend(dataset[split][self.label_column])
        else:
            labels.extend(dataset[self.label_column])

        labels = [item for sublist in labels for item in sublist] if isinstance(labels[0], list) else labels
        return torch.tensor(labels)

    def prepare_sentences(self, dataset: Union[DatasetDict, Dataset]) -> List[str]:
        """Gather sentences from the text column."""
        sentences = []
        if isinstance(dataset, DatasetDict):
            for split in dataset.keys():
                sentences.extend(dataset[split][self.text_column])
        else:
            sentences.extend(dataset[self.text_column])
        return sentences

    @staticmethod
    def _downsample(dataset: DatasetDict, ratio: float) -> DatasetDict:
        """Down-sample the dataset to the specified ratio."""
        for split in dataset.keys():
            dataset[split] = dataset[split].shuffle(seed=42).select(
                range(int(len(dataset[split]) * ratio))
            )
        return dataset

    @staticmethod
    def _find_text_and_label_columns(dataset: DatasetDict, text_column: Optional[str] = None,
                                     label_column: Optional[str] = None) -> Tuple[str, str, type[Any]]:
        """Determine text and label columns in hf datasets based on commonly used keywords"""
        column_names = dataset[next(iter(dataset))].column_names

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

        if not text_column:
            # Iterate over keywords and check if it exists in the dataset
            text_column = next((col for keyword in text_columns for col in column_names if keyword in col), None)
        if not label_column:
            label_column = next((col for keyword in label_columns for col in column_names if keyword in col), None)

        if not text_column or not label_column:
            missing = 'text' if not text_column else 'label'
            raise KeyError(f"Can not determine the {missing} column. Specify {missing}_column=\"...\" "
                           f"from available columns: {column_names}.")

        label_type = type(dataset[next(iter(dataset))][label_column][0])

        return text_column, label_column, label_type

    @staticmethod
    def _merge_textpairs(dataset: DatasetDict, text_column: str, text_pair_column: str) -> DatasetDict:
        """Merge text pairs into a single column with a separator."""
        def merge_texts(example: Dict[str, str]) -> Dict[str, str]:
            example[text_column] = example[text_column] + " [SEP] " + example[text_pair_column]
            return example

        for split in dataset.keys():
            dataset[split] = dataset[split].map(merge_texts, num_proc=None, desc="Merge sentence pair columns")
        return dataset

    @staticmethod
    def _find_task_type(label_column: str, label_type: Union[type(int), type(str), type(list), type(float)]) -> str:
        """Determine the task type based on the label column's data type."""
        label_type_to_task_type = {
            int: "sentence classification",  # labels can be integers e.g. "1"
            str: "sentence classification",  # or strings e.g. "positive"
            list: "word classification",
            float: "sentence regression",
        }

        task_type = label_type_to_task_type.get(label_type, None)

        if not task_type:
            raise ValueError(f"Unable to determine task type from the label column '{label_column}' "
                             f"value: {type(label_type)}.")
        return task_type

    @staticmethod
    def _tokenize(dataset: DatasetDict, pre_tokenizer: Whitespace, text_column: str) -> DatasetDict:
        """Tokenize the dataset using the specified hf pre-tokenizer (e.g. Whitespace)"""
        def pre_tokenize(example):
            encoding = pre_tokenizer.pre_tokenize_str(example[text_column])
            example[text_column] = [token for token, _ in encoding]
            return example

        for split in dataset.keys():
            dataset[split] = dataset[split].map(pre_tokenize, num_proc=None,
                                                desc="Pre-tokenizing texts using Whitespace")
        return dataset

    @staticmethod
    def _merge_data_splits(dataset: DatasetDict) -> Dataset:
        """Merge data splits into a single dataset."""
        datasets_to_merge = [dataset[split] for split in dataset.keys()]
        merged_dataset = datasets.concatenate_datasets(datasets_to_merge)
        return merged_dataset

    @staticmethod
    def _remove_empty_rows(dataset: Union[DatasetDict, Dataset], text_column: str, label_column: str) -> Union[
        DatasetDict, Dataset]:
        """Remove entries with empty sentences or labels."""
        def dataset_row_is_clean(example) -> bool:
            text = example[text_column]
            label = example[label_column]
            entry_has_text = bool(text)  # Check if text is not empty
            label_is_not_none = label is not None  # Check if label is not None
            label_is_valid = label_is_not_none and (isinstance(label, int) and label >= 0)
            keep_entry = entry_has_text and label_is_valid  # keep entries that have text and labels
            return keep_entry

        if isinstance(dataset, DatasetDict):
            for split in dataset.keys():
                dataset[split] = dataset[split].filter(dataset_row_is_clean, desc="Removing empty sentences")
        else:
            dataset = dataset.filter(dataset_row_is_clean, desc="Removing empty sentences")

        return dataset

    @staticmethod
    def _remove_columns(dataset: Union[DatasetDict, Dataset], keep_columns: List[str]) -> Union[DatasetDict, Dataset]:
        """Remove columns from the dataset that are not in the keep_columns list
        (e.g. remove all columns except the text and the label column)"""
        if isinstance(dataset, DatasetDict):
            for split in dataset.keys():
                columns_to_remove = [col for col in dataset[split].column_names if col not in keep_columns]
                dataset[split] = dataset[split].remove_columns(columns_to_remove)
        else:
            columns_to_remove = [col for col in dataset.column_names if col not in keep_columns]
            dataset = dataset.remove_columns(columns_to_remove)
        return dataset

    @staticmethod
    def _make_labels_categorical(
            dataset: Union[DatasetDict, Dataset],
            label_column: str
    ) -> Tuple[Union[DatasetDict, Dataset], Dict[str, int]]:
        """Convert string labels to categorical integer labels for sentence classification tasks."""
        unique_labels = set()
        if isinstance(dataset, DatasetDict):
            for split in dataset.keys():
                unique_labels.update(dataset[split][label_column])
        else:
            unique_labels.update(dataset[label_column])

        label_map = {str(label): idx for idx, label in enumerate(sorted(unique_labels))}

        def map_labels(example):
            example[label_column] = label_map[example[label_column]]
            return example

        if isinstance(dataset, DatasetDict):
            for split in dataset.keys():
                dataset[split] = dataset[split].map(map_labels, num_proc=None,
                                                    desc="Mapping string labels to integers")
        else:
            dataset = dataset.map(map_labels, num_proc=None, desc="Mapping string labels to integers")

        return dataset, label_map

    @staticmethod
    def _change_to_span_encoding(
        dataset: Union[DatasetDict, Dataset],
        label_column: str,
        label_map: Optional[Dict[str, int]] = None,
    ) -> Tuple[Union[DatasetDict, Dataset], Dict[str, int]]:
        """Convert BIO encoding labels to span encoding and update the label map.

        This method transforms labels from BIO (Beginning, Inside, Outside) encoding
        to a simpler span encoding by removing the BIO prefix. It also updates the
        label map accordingly.

        Original BIO labels: ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC']
        Converted span labels: ['PER', 'PER', 'O', 'LOC', 'LOC']

        Original label map: {'B-PER': 0, 'I-PER': 1, 'O': 2, 'B-LOC': 3, 'I-LOC': 4}
        Converted span label map: {'PER': 0, 'O': 1, 'LOC': 2}

        Returns:
        - The dataset with label column converted to span encoding.
        - The updated span label map.
        """
        # Try to retrieve the label map from dataset features
        if not label_map:
            features = dataset[next(iter(dataset))].features if isinstance(dataset, DatasetDict) else dataset.features

            if label_column in features and hasattr(features[label_column], 'feature') and hasattr(
                    features[label_column].feature, 'names'):
                id2label = features[label_column].feature.names
                label_map = {name: idx for idx, name in enumerate(id2label)}
            else:
                # Try to create label map manually if not found in the dataset features
                logger.info(f'Label map was not found in the original dataset. Trying to create label map manually...')
                unique_labels = set()
                for split in dataset.keys() if isinstance(dataset, DatasetDict) else [dataset]:
                    for label_list in dataset[split][label_column] if isinstance(dataset, DatasetDict) else dataset[label_column]:
                        for label in label_list:
                            main_label = label.split('-')[-1] if isinstance(label, str) else str(label)
                            unique_labels.add(main_label)
                unique_labels = sorted(unique_labels, key=int)
                label_map = {label: idx for idx, label in enumerate(unique_labels)}

        logger.info(f"Label map: {label_map}")

        # Modify the label map to remove BIO encoding
        span_label_map = {}
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

    def log_dataset_info(self) -> None:
        """Log information about the dataset after cleaning it"""
        logger.info(f"Text and label columns: '{self.text_column}', '{self.label_column}'")
        logger.info(f"Task type: '{self.task_type}'")
        downsample_info = f"(downsampled to {self.dataset_downsample})" if self.dataset_downsample else ""
        logger.info(f"Dataset size: {self.dataset_size} {downsample_info}")
