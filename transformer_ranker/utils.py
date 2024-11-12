import logging
import operator
import warnings
from typing import Dict, List

from transformers import logging as transformers_logging


def prepare_popular_models(model_size="base") -> List[str]:
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

    return large_models if model_size == "large" else base_models


def configure_logger(
    name: str, level: int = logging.INFO, log_to_console: bool = True
) -> logging.Logger:
    """
    Configure transformer-ranker logger.

    :param name: The name of the logger.
    :param level: The logging level (default: logging.INFO).
    :param log_to_console: Whether to log to console (default: True)
    :return: Configured TransformerRanker logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers and log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter("transformer_ranker:%(message)s"))
        logger.addHandler(console_handler)

    # Ignore specific warning messages from transformers and datasets libraries
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)
    transformers_logging.set_verbosity_error()

    # Suppress transformers warning about unused prediction head weights if the model is frozen
    logger.addFilter(
        lambda record: not (
            "Some weights of BertModel were not initialized" in record.getMessage()
            or "You should probably TRAIN this model" in record.getMessage()
        )
    )

    logger.propagate = False
    return logger


class Result:
    def __init__(self, metric: str):
        """Store all rankings and transferability scores in Result.
        Includes scores for each layer in "layer_estimates".

        param metric: metric name (e.g. "hscore", or "logme")
        """
        self.metric = metric
        self._results: Dict[str, float] = {}
        self.layerwise_scores: Dict[str, Dict[int, float]] = {}

    @property
    def results(self) -> Dict[str, float]:
        """Return the result dictionary sorted by scores in descending order"""
        return dict(sorted(self._results.items(), key=lambda x: x[1], reverse=True))

    @property
    def best_model(self) -> str:
        """Return the highest scoring model"""
        model_name, _ = max(self.results.items(), key=lambda item: item[1])
        return model_name

    @property
    def top_three(self) -> Dict[str, float]:
        """Return three highest scoring models"""
        return {k: self.results[k] for k in list(self.results.keys())[: min(3, len(self.results))]}

    @property
    def best_layers(self) -> Dict[str, int]:
        """Return a dictionary mapping each model name to its best layer ID."""
        best_layers_dict = {}
        for model, values in self.layerwise_scores.items():
            best_layer = max(values.items(), key=operator.itemgetter(1))[0]
            best_layers_dict[model] = best_layer
        return best_layers_dict

    def add_score(self, model_name, score) -> None:
        self._results[model_name] = score

    def append(self, additional_results: "Result") -> None:
        if isinstance(additional_results, Result):
            self._results.update(additional_results.results)
            self.layerwise_scores.update(additional_results.layerwise_scores)
        else:
            raise ValueError(
                f"Expected an instance of 'Result', but got {type(additional_results).__name__}. "
                f"Only 'Result' instances can be appended."
            )

    def _format_results(self) -> str:
        """Helper method to return sorted results as a formatted string."""
        sorted_results = sorted(self._results.items(), key=lambda item: item[1], reverse=True)
        result_lines = [
            f"Rank {i + 1}. {model_name}: {score}"
            for i, (model_name, score) in enumerate(sorted_results)
        ]
        return "\n".join(result_lines)

    def __str__(self) -> str:
        """Return sorted results as a string (user-friendly)."""
        return self._format_results()

    def __repr__(self) -> str:
        """Return sorted results as a string (coder-friendly)."""
        return self._format_results()
