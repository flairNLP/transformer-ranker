import logging
import operator
import warnings

from transformers import logging as transformers_logging


def prepare_popular_models(model_size="base") -> list[str]:
    """Two lists of pretrained models to try out"""

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


def configure_logger(name: str, level: int = logging.INFO, log_to_console: bool = True) -> logging.Logger:
    """
    Configure logger for this transferability NLP framework

    :param name: The name of the logger "transformer-ranker".
    :param level: The logging level (default: logging.INFO).
    :param log_to_console: Whether to log to console (default: True)
    :return: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers and log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter("transformer_ranker:%(message)s"))
        logger.addHandler(console_handler)

    # Suppress future and user warnings 
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)
    transformers_logging.set_verbosity_error()

    # Suppress unused weights messages when loading models (transformers)
    logger.addFilter(
        lambda record: not (
            "Some weights of BertModel were not initialized" in record.getMessage()
            or "You should probably TRAIN this model" in record.getMessage()
        )
    )

    logger.propagate = False
    return logger


class Result:
    """Store model names and transferability scores in a dictionary-like result"""

    def __init__(self, metric: str):
        self.metric = metric
        self.scores = {}
        self.layerwise_scores = {}

    def add_score(self, model_name: str, score: float) -> None:
        self.scores[model_name] = score

    def add_layerwise_scores(self, model_name: str, scores: dict) -> None:
        """Only used for the bestlayer option."""
        self.layerwise_scores[model_name] = scores

    def append(self, other: "Result") -> None:
        """Append scores from multiple runs."""
        if self.metric != other.metric:
            raise ValueError(f"Metrics do not match ({self.metric} vs {other.metric}).")
        self.scores.update(other.scores)
        self.layerwise_scores.update(other.layerwise_scores)

    def best_model(self) -> str:
        """Show the model with the highest score."""
        if not self.scores:
            return ""
        return max(self.scores, key=self.scores.get)

    def __str__(self) -> str:
        """Sort scores and print them with rankings."""
        if not self.scores:
            return "No scores available."

        sorted_scores = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        model_rank = [f"Rank {i+1}. {model}: {score:.4f}" for i, (model, score) in enumerate(sorted_scores)]
        return "\n".join(model_rank)

    def __repr__(self) -> str:
        return self.__str__()
