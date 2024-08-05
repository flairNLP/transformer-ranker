import torch
from datasets.dataset_dict import Dataset, DatasetDict
from tqdm import tqdm

from .embedder import Embedder
from .estimators import HScore, LogME, KNN
from .utils import DatasetCleaner, Result, configure_logger

import logging
import warnings
from typing import List, Optional, Union


# Ignore specific warning messages from transformers and datasets libraries (it's harmless but annoying)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

logger = configure_logger('transformer_ranker', logging.INFO)


class TransformerRanker:
    def __init__(
        self,
        dataset: Union[str, Dataset, DatasetDict],
        dataset_downsample: Optional[float] = None,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        task_type: Optional[str] = None,
        **kwargs
    ):
        """
        Rank transformer models based on their estimated performance for a specific NLP task.
        To rank models we use transferability estimation methods e.g. h-score or logme and take features from
        deeper layers by averaging all layers or by searching the best performing layer in each model.

        :param dataset: huggingface dataset for evaluating transformer models, containing texts and label columns.
        :param dataset_downsample: a fraction to which the dataset should be down-sampled.
        :param kwargs: Additional parameters for data pre-processing.
        """
        # Instantiate the data pre-processing class with specified parameters
        self.data_handler = DatasetCleaner(dataset_downsample=dataset_downsample,
                                           text_column=text_column,
                                           label_column=label_column,
                                           task_type=task_type,
                                           **kwargs,
                                           )

        # Pre-process a huggingface dataset to be used by the ranker
        self.dataset = self.data_handler.prepare_dataset(dataset)

        # Determine task type if not specified: word classification or text classification
        self.task_type = self.data_handler.task_type

        # Find text and label columns if not specified
        self.text_column = self.data_handler.text_column
        self.label_column = self.data_handler.label_column

    def run(
        self,
        models: List[Union[str, torch.nn.Module]],
        batch_size: int = 32,
        estimator: str = "hscore",
        layer_aggregator: str = "layermean",
        sentence_pooling: str = "mean",
        device: Optional[str] = None,
        gpu_estimation: bool = True,
        **kwargs
    ):
        """
        Run the ranker which embeds documents using each model and estimates the transferability of embeddings.

        :param models: A list of identifiers for the transformer models to be evaluated.
        :param batch_size: The number of samples to process in each batch, defaults to 32.
        :param estimator: Approach to assess model performance (e.g., 'hscore', 'logme', 'knn').
        :param layer_aggregator: How to combine outputs from different layers (e.g., 'layermean', 'bestlayer').
        :param sentence_pooling: Parameter for embedder class, telling how to pool words into a text embedding for
        text classification tasks. Defaults to "mean", which averaged of all words into a single text embedding.
        :param device: Device used to embed, defaults to gpu if available (e.g. 'cpu', 'cuda', 'cuda:2').
        :param gpu_estimation: If to store embeddings on gpu and run estimation using gpu for speedup.
        :param kwargs: Additional parameters for the embedder class (e.g. subword-pooling)
        :return: Returns the sorted dictionary of transformer handles and their transferability scores
        """
        self._validate_ranker_setup(estimator=estimator, layer_aggregator=layer_aggregator)

        result_dictionary = Result(metric=estimator)

        # Load all transformers into hf cache for later use
        self._preload_transformers(models)

        device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        labels = self.data_handler.prepare_labels(self.dataset)
        labels = labels.to(device) if gpu_estimation else labels

        # Iterate over each transformer model and score it
        for model_id, model_name in enumerate(models):

            # Set which transformer layers will be used: last layer (i.e. output layer) or all of the layers
            layer_ids = "-1" if layer_aggregator == "lastlayer" else "all"
            layer_pooling = "mean" if "mean" in layer_aggregator else None

            # Sentence pooling is only applied for text classification tasks
            sentence_pooling = None if self.task_type == "word classification" else sentence_pooling

            embedder = Embedder(
                model=model_name,
                layer_ids=layer_ids,
                layer_pooling=layer_pooling,
                sentence_pooling=sentence_pooling,
                device=device,
                **kwargs
            )

            embeddings = embedder.embed(
                self.data_handler.prepare_sentences(self.dataset),
                batch_size=batch_size,
                show_loading_bar=True,
                move_embeddings_to_cpu=False if gpu_estimation else True,
            )

            # Flatten embeddings to have a list of all word embeddings for sequence tagging tasks
            if self.task_type == "word classification":
                embeddings = [word_embedding for sentence_embedding in embeddings
                              for word_embedding in sentence_embedding]

            layer_ids = embedder.layer_ids
            num_layers = embeddings[0].size(0)
            layer_scores = []

            # Remove transformer model from memory after embeddings are extracted
            del embedder

            # Estimate scores for each layer
            tqdm_bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
            for layer_id in tqdm(range(num_layers), desc="Estimating Performance", bar_format=tqdm_bar_format):
                # Get the position of the layer index
                layer_index = layer_ids[layer_id]

                # Stack embeddings for that layer
                layer_embeddings = torch.stack([word_embedding[layer_index] for word_embedding in embeddings])

                # Estimate score using layer embeddings and ground truth labels
                score = self._estimate_score(estimator=estimator,
                                             embeddings=layer_embeddings,
                                             labels=labels,
                                             )
                layer_scores.append(score)

            # Store scores for each layer in the result dictionary
            result_dictionary.layer_estimates[model_name] = dict(zip(layer_ids, layer_scores))

            # Aggregate scores for each layer
            if layer_aggregator in ["layermean", "lastlayer"]:
                final_score = layer_scores[0]
            elif layer_aggregator == "max_of_scores":
                final_score = max(layer_scores)
            else:  # self.layer_aggregator == "avg_of_scores"
                final_score = sum(layer_scores) / len(layer_scores)

            result_dictionary.add_score(model_name, final_score)

            # Log the scoring information for a model
            base_log = f"{model_name}, estimated score: {final_score}"
            layer_estimates_log = (f", layer estimates: {result_dictionary.layer_estimates[model_name]}"
                                   if layer_aggregator == 'bestlayer' else "")
            logger.info(base_log + layer_estimates_log)

        return result_dictionary

    @staticmethod
    def _preload_transformers(models: List[Union[str, torch.nn.Module]]) -> None:
        """Loads all models into huggingface cache"""
        for model_name in models:
            Embedder(model_name)

    def _validate_ranker_setup(self, estimator, layer_aggregator) -> None:
        """Validate if estimator, aggregator and task type are used correctly"""
        valid_estimators = ["hscore", "logme", "knn"]
        if estimator not in valid_estimators:
            raise ValueError(f"Unsupported estimation method: {estimator}. "
                             f"Use one of the following {valid_estimators}")

        valid_layer_aggregators = ["layermean", "lastlayer", "max_of_scores", "average_of_scores"]
        if layer_aggregator not in valid_layer_aggregators:
            raise ValueError(f"Unsupported layer pooling: {layer_aggregator}. "
                             f"Use one of the following {valid_layer_aggregators}")

        valid_task_types = ["sentence classification", "word classification", "sentence regression"]
        if self.task_type not in valid_task_types:
            raise ValueError("Unable to determine task type of the dataset. Please specify it as a parameter: "
                             "task_type= \"sentence classification\", \"sentence regression\", or "
                             "\"word classification\"")

    def _estimate_score(self, estimator, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
        """Use the specified estimator to score a transformer"""
        regression = self.task_type == "sentence regression"
        if estimator == 'hscore' and regression:
            logger.warning(f'Specified estimator="{estimator}" does not support regression tasks.'
                           f'We recommend using LogME for regression (estimator="logme")')

        estimator_classes = {
            "knn": KNN(k=3, regression=regression),
            "hscore": HScore(),
            "logme": LogME(regression=regression)
        }

        estimator = estimator_classes[estimator]
        score = estimator.fit(features=embeddings, labels=labels)

        return round(score, 4)
