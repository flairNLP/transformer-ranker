import logging
from typing import Any, Optional, Union

import torch
from datasets.dataset_dict import Dataset, DatasetDict
from tqdm import tqdm

from .datacleaner import DatasetCleaner, TaskCategory
from .embedder import Embedder
from .estimators import HScore, LogME, NearestNeighbors
from .utils import Result, configure_logger

logger = configure_logger("transformer_ranker", logging.INFO)


class TransformerRanker:
    def __init__(
        self,
        dataset: Union[str, Dataset, DatasetDict],
        dataset_downsample: Optional[float] = None,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Prepares a dataset and compiles metrics to assess transferability.

        :param dataset: a dataset from huggingface with texts and labels.
        :param dataset_downsample: a fraction to which the dataset should be reduced.
        :param text_column: the name of the column containing texts.
        :param label_column: the name of the column containing labels.
        :param kwargs: additional dataset-specific parameters for data cleaning.
        """
        # Preprocess and down-sample a dataset
        datacleaner = DatasetCleaner(
            dataset_downsample=dataset_downsample,
            text_column=text_column,
            label_column=label_column,
            **kwargs,
        )

        self.texts, self.labels, self.task_category = datacleaner.prepare_dataset(dataset)

        # Supported metrics
        self.transferability_metrics = {
            "logme": LogME,
            "hscore": HScore,
            "knn": NearestNeighbors,
        }

    def run(
        self,
        models: list[Union[str, torch.nn.Module]],
        estimator: str = "hscore",
        layer_aggregator: str = "layermean",
        batch_size: int = 32,
        **kwargs: Any,
    ):
        """
        Loads models, collects embeddings, and scores them.

        :param models: A list of model names
        :param estimator: Transferability metric ('hscore', 'logme', 'knn').
        :param layer_aggregator: Method to aggregate layers ('lastlayer', 'layermean', 'bestlayer').
        :param batch_size: Number of samples per batch, defaults to 32.
        :param device: Device for embedding ('cpu', 'cuda', 'cuda:2').
        :param gpu_estimation: Store and score embeddings on GPU for speedup.
        :param kwargs: Additional parameters for embedder class.
        :return: Returns sorted dictionary of model names and their scores
        """
        self._confirm_ranker_setup(estimator=estimator, layer_aggregator=layer_aggregator)

        # Set device for models and the metric
        device = kwargs.pop("device", None)
        gpu_estimation = kwargs.get("gpu_estimation", True)
        if gpu_estimation:
            self.labels = self.labels.to(device)

        # Download models to hf cache
        self._preload_models(models=models, device=device)

        # Set transferability metric
        regression = self.task_category == TaskCategory.TEXT_REGRESSION
        metric = self.transferability_metrics[estimator](regression=regression)
        result = Result(metric=estimator)

        for model in models:
            effective_sentence_pooling = (
                None
                if self.task_category == TaskCategory.TOKEN_CLASSIFICATION
                else kwargs.get("sentence_pooling", "mean")
            )

            # Setup the embedder
            embedder = Embedder(
                model=model,
                layer_ids="0" if layer_aggregator == "lastlayer" else "all",
                layer_mean=True if "mean" in layer_aggregator else False,
                sentence_pooling=effective_sentence_pooling,
                device=device,
                **kwargs,
            )

            # Collect embeddings
            embeddings = embedder.embed(
                self.texts, batch_size=batch_size, unpack_to_cpu=not gpu_estimation, show_progress=True,
            )  # fmt: skip

            # Flatten them for ner tasks
            if self.task_category == TaskCategory.TOKEN_CLASSIFICATION:
                embeddings = [word for sentence in embeddings for word in sentence]

            model_name = embedder.name
            del embedder  # remove from memory
            torch.cuda.empty_cache()

            # Compute transferability
            score = self._transferability_score(embeddings, metric, layer_aggregator)

            # Store and log results
            result.add_score(model_name, score)
            logger.info(f"{model_name} {result.metric}: {score:.4f}")

        return result

    def _transferability_score(self, embeddings, metric, layer_aggregator, show_progress=True) -> float:
        """Compute transferability for a model"""
        tqdm_bar_format = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
        num_layers, scores_per_layer = len(embeddings[0]), []

        transferability_progress = tqdm(
            range(num_layers), desc="Transferability score", bar_format=tqdm_bar_format, disable=not show_progress
        )

        # Score each layer separately
        for layer_id in transferability_progress:
            layer_embeddings = torch.stack([emb[layer_id] for emb in embeddings])
            score = metric.fit(embeddings=layer_embeddings, labels=self.labels)
            scores_per_layer.append(score)

        aggregated_score = (
            max(scores_per_layer) if layer_aggregator == "bestlayer" else scores_per_layer[-1]
        )  # fmt :skip

        return aggregated_score

    def _preload_models(self, models: list[str], device: Optional[str] = None) -> None:
        """Load models to HuggingFace cache if not already present"""
        cached_models, downloaded_models = set(), set()

        for model in models:
            try:
                Embedder(model, local_files_only=True, device=device)
                cached_models.add(model)
            except (OSError, RuntimeError):
                downloaded_models.add(model)

        if cached_models:
            logger.info(f"Models found in cache: {cached_models}")

        if downloaded_models:
            logger.info(f"Downloading models: {downloaded_models}")

            for model in downloaded_models:
                Embedder(model, device=device)

    def _confirm_ranker_setup(self, estimator: str, layer_aggregator: str) -> None:
        """Validate main parameters in the run method"""
        available_metrics = self.transferability_metrics.keys()
        if estimator not in available_metrics:
            raise ValueError(f"Unsupported metric '{estimator}'. Valid options: {', '.join(available_metrics)}")

        available_layer_pooling = {"layermean", "lastlayer", "bestlayer"}
        if layer_aggregator not in available_layer_pooling:
            raise ValueError(
                f"Unsupported aggregation '{layer_aggregator}'. Valid options: {', '.join(available_layer_pooling)}"
            )
