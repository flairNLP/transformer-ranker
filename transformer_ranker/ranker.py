import logging
from typing import Any, Optional, Union

import torch
from datasets.dataset_dict import Dataset, DatasetDict
from tqdm import tqdm

from .datacleaner import DatasetCleaner, TaskCategory
from .embedder import Embedder
from .estimators import KNN, HScore, LogME
from .utils import Result, configure_logger

logger = configure_logger("transformer_ranker", logging.INFO)


class TransformerRanker:
    def __init__(
        self,
        dataset: Union[str, Dataset, DatasetDict],
        dataset_downsample: Optional[float] = None,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Rank language models for different NLP tasks. Embed a part of the dataset and
        estimate embedding suitability with transferability metrics like hscore or logme.
        Embeddings can be averaged across all layers or selected from the best-performing layer.

        :param dataset: a dataset from huggingface, containing texts and label columns.
        :param dataset_downsample: a fraction to which the dataset should be reduced.
        :param kwargs: Additional dataset-specific parameters for data cleaning.
        """
        # Clean the original dataset and keep only needed columns
        datacleaner = DatasetCleaner(
            dataset_downsample=dataset_downsample,
            text_column=text_column,
            label_column=label_column,
            **kwargs,
        )

        self.texts: Union[list[str], list[list[str]]]
        self.labels: torch.Tensor
        self.task_category: TaskCategory

        self.texts, self.labels, self.task_category = datacleaner.prepare_dataset(dataset)

    def run(
        self,
        models: list[Union[str, torch.nn.Module]],
        batch_size: int = 32,
        estimator: str = "hscore",
        layer_aggregator: str = "layermean",
        sentence_pooling: str = "mean",
        device: Optional[str] = None,
        gpu_estimation: bool = True,
        **kwargs: Any
    ):
        """
        Load models, get embeddings, score, and rank results.

        :param models: A list of model names string identifiers
        :param batch_size: The number of samples to process in each batch, defaults to 32.
        :param estimator: Transferability metric (e.g., 'hscore', 'logme', 'knn').
        :param layer_aggregator: Which layer to select (e.g., 'layermean', 'bestlayer').
        :param sentence_pooling: Embedder parameter for pooling words into a sentence embedding for
        text classification tasks. Defaults to "mean" to average of all words.
        :param device: Device for embedding, defaults to GPU if available ('cpu', 'cuda', 'cuda:2').
        :param gpu_estimation: Store and score embeddings on GPU for speedup.
        :param kwargs: Additional parameters for the embedder class (e.g. subword pooling)
        :return: Returns the sorted dictionary of model names and their scores
        """
        self._confirm_ranker_setup(estimator=estimator, layer_aggregator=layer_aggregator)

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load all transformers into hf cache
        self._preload_transformers(models, device)

        # Device for transferability estimation
        if gpu_estimation:
            self.labels = self.labels.to(device)

        # Store all results in a dictionary
        ranking_results = Result(metric=estimator)

        # Iterate over each transformer model and score it
        for model in models:

            # Select transformer layers: last layer or all layers
            layer_ids = "-1" if layer_aggregator == "lastlayer" else "all"
            layer_pooling = "mean" if "mean" in layer_aggregator else None

            # Set effective sentence pooling as parameter
            effective_sentence_pooling = (
                None if self.task_category == TaskCategory.TOKEN_CLASSIFICATION else sentence_pooling
            )

            # Prepare the transformer embedder with word, sentence, and layer pooling
            embedder = Embedder(
                model=model,
                layer_ids=layer_ids,
                layer_pooling=layer_pooling,
                sentence_pooling=effective_sentence_pooling,
                device=device,
                **kwargs,
            )

            # Gather models embeddings for the dataset
            embeddings = embedder.embed(
                self.texts,
                batch_size=batch_size,
                show_loading_bar=True,
                move_embeddings_to_cpu=not gpu_estimation,
            )

            # Transferability for word tasks: all word embeddings in one list
            if self.task_category == TaskCategory.TOKEN_CLASSIFICATION:
                embeddings = [word for sentence in embeddings for word in sentence]

            model_name = embedder.model_name
            embedded_layer_ids = embedder.layer_ids
            num_layers = embeddings[0].size(0)

            # Remove transformer model from memory after embeddings are extracted
            del embedder
            torch.cuda.empty_cache()

            # Estimate scores for each layer
            layer_scores = []
            tqdm_bar_format = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
            for layer_id in tqdm(
                range(num_layers), desc="Transferability Score", bar_format=tqdm_bar_format
            ):
                # Get the position of the layer index
                layer_index = embedded_layer_ids[layer_id]

                # Stack embeddings for that layer
                layer_embeddings = torch.stack(
                    [word_embedding[layer_index] for word_embedding in embeddings]
                )

                # Estimate score using layer embeddings and labels
                score = self._estimate_score(
                    estimator=estimator,
                    embeddings=layer_embeddings,
                    labels=self.labels,
                )
                layer_scores.append(score)

            # Store scores for each layer in the result dictionary
            ranking_results.layerwise_scores[model_name] = dict(
                zip(embedded_layer_ids, layer_scores)
            )

            # Aggregate layer scores
            final_score = max(layer_scores) if layer_aggregator == "bestlayer" else layer_scores[0]
            ranking_results.add_score(model_name, final_score)

            # Log the final score along with scores for each layer
            result_log = f"{model_name} estimation: {final_score} ({ranking_results.metric})"

            # Log scores for layer ranking
            if layer_aggregator == "bestlayer":
                result_log += f", scores for each layer: {ranking_results.layerwise_scores[model_name]}"

            logger.info(result_log)

        return ranking_results

    @staticmethod
    def _preload_transformers(
        models: list[Union[str, torch.nn.Module]], device: Optional[str] = None
    ) -> None:
        """Loads all models into HuggingFace cache"""
        cached_models, download_models = [], []
        for model_name in models:
            try:
                Embedder(model_name, local_files_only=True, device=device)
                cached_models.append(model_name)
            except OSError:
                download_models.append(model_name)

        logger.info(f"Models found in cache: {cached_models}") if cached_models else None
        logger.info(f"Downloading models: {download_models}") if download_models else None

        for model_name in models:
            Embedder(model_name, device=device)

    def _confirm_ranker_setup(self, estimator, layer_aggregator) -> None:
        """Validate estimator and layer selection setup"""
        valid_estimators = ["hscore", "logme", "knn"]
        if estimator not in valid_estimators:
            raise ValueError(
                f"Unsupported estimation method: {estimator}. "
                f"Use one of the following {valid_estimators}"
            )

        valid_layer_aggregators = ["layermean", "lastlayer", "bestlayer"]
        if layer_aggregator not in valid_layer_aggregators:
            raise ValueError(
                f"Unsupported layer pooling: {layer_aggregator}. "
                f"Use one of the following {valid_layer_aggregators}"
            )

        valid_task_categories = ["text classification", "token classification", "text regression"]
        if self.task_category not in valid_task_categories:
            raise ValueError(
                "Unable to determine task type of the dataset. Please specify it as a parameter: "
                'task_category= "text classification", "token classification", or '
                '"text regression"'
            )

        if self.task_category == TaskCategory.TEXT_REGRESSION and estimator == "hscore":
            supported_estimators = [est for est in valid_estimators if est != "hscore"]
            raise ValueError(
                f'"{estimator}" does not support text regression. '
                f"Use one of the following estimators: {supported_estimators}"
            )

    def _estimate_score(self, estimator, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
        """Use an estimator to score a transformer"""
        regression = self.task_category == TaskCategory.TEXT_REGRESSION

        estimator_classes = {
            "knn": KNN(k=3, regression=regression),
            "logme": LogME(regression=regression),
            "hscore": HScore(),
        }

        estimator = estimator_classes[estimator]
        score = estimator.fit(embeddings=embeddings, labels=labels)

        return round(score, 4)
