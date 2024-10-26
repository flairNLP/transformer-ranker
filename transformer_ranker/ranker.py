import logging
from typing import List, Optional, Union

import torch
from datasets.dataset_dict import Dataset, DatasetDict
from tqdm import tqdm

from .datacleaner import DatasetCleaner
from .embedder import Embedder
from .estimators import KNN, HScore, LogME
from .utils import Result, configure_logger

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
        Rank language models based on their predicted performance for a specific NLP task.
        We use metrics like h-score or logme to estimate the quality of embeddings. Features are taken from
        deeper layers by averaging all layers or by selecting the best-scoring layer in each model.

        :param dataset: huggingface dataset for evaluating transformer models, containing texts and label columns.
        :param dataset_downsample: a fraction to which the dataset should be down-sampled.
        :param kwargs: Additional parameters for data pre-processing.
        """
        # Clean the original dataset and keep only needed columns
        self.data_handler = DatasetCleaner(dataset_downsample=dataset_downsample,
                                           text_column=text_column,
                                           label_column=label_column,
                                           task_type=task_type,
                                           **kwargs,
                                           )

        self.dataset = self.data_handler.prepare_dataset(dataset)

        # Find task type if not given: word classification or text classification
        self.task_type = self.data_handler.task_type

        # Find text and label columns
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
        The run method loads the models, gathers embeddings for each, scores them, and sorts the results to rank them.

        :param models: A list of model names string identifiers
        :param batch_size: The number of samples to process in each batch, defaults to 32.
        :param estimator: A metric to assess model performance (e.g., 'hscore', 'logme', 'knn').
        :param layer_aggregator: Which layer to select (e.g., 'layermean', 'bestlayer').
        :param sentence_pooling: Parameter for embedder class, telling how to pool words into a sentence embedding for
        text classification tasks. Defaults to "mean" to average of all words.
        :param device: Device used to embed, defaults to gpu if available (e.g. 'cpu', 'cuda', 'cuda:2').
        :param gpu_estimation: If to store embeddings on gpu and run estimation using gpu for speedup.
        :param kwargs: Additional parameters for the embedder class (e.g. subword-pooling)
        :return: Returns the sorted dictionary of model names and their scores
        """
        self._confirm_ranker_setup(estimator=estimator, layer_aggregator=layer_aggregator)

        # Load all transformers into hf cache for later use
        self._preload_transformers(models)

        labels = self.data_handler.prepare_labels(self.dataset)

        result_dictionary = Result(metric=estimator)

        # Iterate over each transformer model and score it
        for model in models:

            # Select transformer layers to be used: last layer (i.e. output layer) or all of the layers
            layer_ids = "-1" if layer_aggregator == "lastlayer" else "all"
            layer_pooling = "mean" if "mean" in layer_aggregator else None

            # Sentence pooling is only applied for text classification tasks
            effective_sentence_pooling = None if self.task_type == "token classification" else sentence_pooling

            embedder = Embedder(
                model=model,
                layer_ids=layer_ids,
                layer_pooling=layer_pooling,
                sentence_pooling=effective_sentence_pooling,
                device=device,
                **kwargs
            )

            embeddings = embedder.embed(
                self.data_handler.prepare_sentences(self.dataset),
                batch_size=batch_size,
                show_loading_bar=True,
                move_embeddings_to_cpu=False if gpu_estimation else True,
            )

            # Single list of embeddings for sequence tagging tasks
            if self.task_type == "token classification":
                embeddings = [word_embedding for sentence_embedding in embeddings
                              for word_embedding in sentence_embedding]

            embedded_layer_ids = embedder.layer_ids
            model_name = embedder.model_name
            num_layers = embeddings[0].size(0)
            layer_scores = []

            if gpu_estimation:
                labels = labels.to(embedder.device)

            # Remove transformer model from memory after embeddings are extracted
            del embedder
            torch.cuda.empty_cache()

            # Estimate scores for each layer
            tqdm_bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
            for layer_id in tqdm(range(num_layers), desc="Estimating Performance", bar_format=tqdm_bar_format):
                # Get the position of the layer index
                layer_index = embedded_layer_ids[layer_id]

                # Stack embeddings for that layer
                layer_embeddings = torch.stack([word_embedding[layer_index] for word_embedding in embeddings])

                # Estimate score using layer embeddings and labels
                score = self._estimate_score(estimator=estimator,
                                             embeddings=layer_embeddings,
                                             labels=labels,
                                             )
                layer_scores.append(score)

            # Store scores for each layer in the result dictionary
            result_dictionary.layer_estimates[model_name] = dict(zip(embedded_layer_ids, layer_scores))

            # Aggregate scores for each layer
            if layer_aggregator in ["layermean", "lastlayer"]:
                final_score = layer_scores[0]
            elif layer_aggregator == "bestlayer":
                final_score = max(layer_scores)
            else:
                logger.warning(f'Unknown estimator: "{estimator}"')
                final_score = 0.

            result_dictionary.add_score(model_name, final_score)

            # Log the scoring information for a model
            base_log = f"{model_name}, score: {final_score}"
            layer_estimates_log = (f", layerwise scores: {result_dictionary.layer_estimates[model_name]}"
                                   if layer_aggregator == 'bestlayer' else "")
            logger.info(base_log + layer_estimates_log)

        return result_dictionary

    @staticmethod
    def _preload_transformers(models: List[Union[str, torch.nn.Module]]) -> None:
        """Loads all models into HuggingFace cache"""
        cached_models, download_models = [], []

        for model_name in models:
            try:
                Embedder(model_name, local_files_only=True)
                cached_models.append(model_name)
            except OSError:
                download_models.append(model_name)

        logger.info(f"Models found in cache: {cached_models}") if cached_models else None
        logger.info(f"Downloading models: {download_models}") if download_models else None

        for model_name in models:
            Embedder(model_name)

    def _confirm_ranker_setup(self, estimator, layer_aggregator) -> None:
        """Validate estimator and layer selection setup"""
        valid_estimators = ["hscore", "logme", "knn"]
        if estimator not in valid_estimators:
            raise ValueError(f"Unsupported estimation method: {estimator}. "
                             f"Use one of the following {valid_estimators}")

        valid_layer_aggregators = ["layermean", "lastlayer", "bestlayer"]
        if layer_aggregator not in valid_layer_aggregators:
            raise ValueError(f"Unsupported layer pooling: {layer_aggregator}. "
                             f"Use one of the following {valid_layer_aggregators}")

        valid_task_types = ["text classification", "token classification", "text regression"]
        if self.task_type not in valid_task_types:
            raise ValueError("Unable to determine task type of the dataset. Please specify it as a parameter: "
                             "task_type= \"text classification\", \"token classification\", or "
                             "\"text regression\"")

    def _estimate_score(self, estimator, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
        """Use an estimator to score a transformer"""
        regression = self.task_type == "text regression"
        if estimator in ['hscore'] and regression:
            logger.warning(f'Specified estimator="{estimator}" does not support regression tasks.')

        estimator_classes = {
            "knn": KNN(k=3, regression=regression),
            "logme": LogME(regression=regression),
            "hscore": HScore(),
        }

        estimator = estimator_classes[estimator]
        score = estimator.fit(embeddings=embeddings, labels=labels)

        return round(score, 4)
