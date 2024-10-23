import logging
import warnings
from transformers import logging as transformers_logging
import operator

from typing import List, Dict


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

    if not logger.handlers and log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter('transformer_ranker:%(message)s'))
        logger.addHandler(console_handler)

    # Ignore specific warning messages from transformers and datasets libraries
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)
    transformers_logging.set_verbosity_error()

    # Suppress transformers warning about unused prediction head weights if the model is frozen
    logger.addFilter(lambda record: not (
        "Some weights of BertModel were not initialized" in record.getMessage() or
        "You should probably TRAIN this model" in record.getMessage()
    ))

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
        self.layer_estimates: Dict[str, Dict[int, float]] = {}

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
    def best_layers(self) -> Dict[str, int]:
        """Return a dictionary with model name: best layer id"""
        return {model: max(values.items(), key=operator.itemgetter(1))[0] for model, values in self.layer_estimates.items()}

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
