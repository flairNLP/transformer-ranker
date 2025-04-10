import warnings
from typing import Any, Optional, Union

import torch
from tokenizers.pre_tokenizers import PreTokenizer, Whitespace
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast


class Embedder:
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        tokenizer: Optional[Union[str, PreTrainedTokenizerFast]] = None,
        pre_tokenizer: Optional[PreTokenizer] = Whitespace(),
        subword_pooling: str = "mean",
        sentence_pooling: Optional[str] = None,
        layer_ids: str = "all",
        layer_mean: Optional[bool] = None,
        local_files_only: bool = False,
        device: Optional[str] = None,
    ):
        """
        Generates word or text embeddings using a pre-trained model.
        Does sub-word and sequence (sentence) pooling.

        :param model: Model name or instance.
        :param tokenizer: Tokenizer name or instance.
        :param pre_tokenizer: Pre-tokenizer for text preprocessing.
        :param subword_pooling: Method for pooling sub-words ('mean', 'first', 'last').
        :param sentence_pooling: Method for pooling words into a text embedding.
        :param layer_ids: Layers to use ('all', or '-1, -2, -3, ...').
        :param layer_mean: Boolean if to average layers.
        :param local_files_only: Load models locally only.
        :param device: Compute device ('cpu', 'cuda:0', 'cuda:1').
        """
        # Setup model and tokenizer
        self._setup_model(model, local_files_only)
        self._setup_tokenizer(tokenizer)
        self.pre_tokenizer = pre_tokenizer

        # Model details
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        # Set word, sentence pooling options
        self.subword_pooling = subword_pooling
        self.sentence_pooling = sentence_pooling
        self.layer_mean = layer_mean
        self.layer_ids = self._parse_layer_ids(layer_ids)

        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def embed(
        self,
        sentences: Union[str, list[str]],
        batch_size: int = 32,
        unpack_to_cpu: bool = True,
        show_progress: bool = True,
    ) -> list[torch.Tensor]:
        """Prepares texts into batches and embeds a dataset."""

        if isinstance(sentences, str):
            sentences = [sentences]

        if not any(sentences):
            warnings.warn("Input text is empty, cannot generate embeddings.")
            return [torch.empty(0)]

        batches = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]
        progress_bar = tqdm(
            batches,
            desc="Generating embeddings",
            disable=not show_progress,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )

        embeddings = []
        for batch in progress_bar:
            batch_embeddings = self._embed_batch(batch, unpack_to_cpu)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _embed_batch(self, sentences, unpack_to_cpu: bool = True) -> list[torch.Tensor]:
        """Embeds a batch of texts. Embeddings can be moved to cpu or kept on gpu."""
        tokenized = self._tokenize(sentences)
        word_ids = tokenized.pop("word_ids")

        # Do forward pass and get all hidden states
        with torch.no_grad():
            outputs = self.model(**tokenized, output_hidden_states=True)
            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states
            else:
                raise ValueError(f"Failed to get hidden states for model: {self.name}")

        # Exclude the embedding layer (index 0)
        embeddings = torch.stack(hidden_states[1:], dim=0)
        embeddings = embeddings.permute(1, 2, 0, 3)

        # Zero out padded tokens
        embeddings = embeddings * tokenized["attention_mask"].unsqueeze(-1).unsqueeze(-1)

        # Select and average hidden states
        embeddings = embeddings[:, :, self.layer_ids, :]
        if self.layer_mean:
            embeddings = torch.mean(embeddings, dim=2, keepdim=True)

        sentence_embeddings = []
        for subword_embeddings, ids in zip(embeddings, word_ids):
            # Pool sub-words to get word embeddings
            word_embeddings = self._pool_subwords(subword_embeddings, ids)

            # Pool words to get text embedding
            sentence_embedding = self._pool_words(word_embeddings) if self.sentence_pooling else word_embeddings
            sentence_embeddings.append(sentence_embedding)

        if unpack_to_cpu:
            sentence_embeddings = [sentence_embedding.cpu() for sentence_embedding in sentence_embeddings]

        return sentence_embeddings

    def _setup_model(self, model: Union[str, torch.nn.Module], local_files_only: bool) -> None:
        """Initialize a model using AutoModel, support custom models."""
        self.model = (
            model
            if isinstance(model, torch.nn.Module)
            else AutoModel.from_pretrained(model, local_files_only=local_files_only)
        )

        if hasattr(self.model.config, 'is_encoder_decoder') and self.model.config.is_encoder_decoder:
            self.model = self.model.encoder  # remove decoder

        self.name = getattr(self.model.config, "name_or_path", "Unknown Model")

    def _setup_tokenizer(self, tokenizer: Optional[Union[str, PreTrainedTokenizerFast]]) -> None:
        """Initialize tokenizer using AutoTokenizer, support PreTokenizerFast."""
        self.tokenizer = (
            tokenizer
            if isinstance(tokenizer, PreTrainedTokenizerFast)
            else AutoTokenizer.from_pretrained(tokenizer or self.name, add_prefix_space=True)
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _tokenize(self, sentences: Union[list[str], list[list[str]]]) -> dict[str, Any]:
        """Pre-tokenize and tokenize texts."""
        if self.pre_tokenizer and isinstance(sentences[0], str):
            sentences = [
                [word for word, _ in self.pre_tokenizer.pre_tokenize_str(sentence)]
                for sentence in sentences
            ]  # fmt: skip

        max_length = self.tokenizer.model_max_length
        max_length = max_length if max_length < 1000000 else 512
        is_split_into_words = not isinstance(sentences[0], str)

        tokenized = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            is_split_into_words=is_split_into_words,
        )

        # Move inputs to device and prepare word_ids for sub-words
        input_dict = {k: v.to(self.device) for k, v in tokenized.items()}
        input_dict["word_ids"] = [tokenized.word_ids(i) for i in range(len(sentences))]
        return input_dict

    def _parse_layer_ids(self, layer_ids: str) -> list[int]:
        """Parse layer ids from a string. Convert negative ids, remove duplicates, sort"""
        if layer_ids == "all":
            return list(range(self.num_layers))
        layer_ids = [int(number) for number in layer_ids.split(",")]

        if any(layer_id >= self.num_layers or layer_id < -self.num_layers for layer_id in layer_ids):
            raise ValueError(f"Layer ids must be within the range of (0 to {self.num_layers - 1}).")

        layer_ids = set(layer_id % self.num_layers for layer_id in layer_ids)
        return sorted(layer_ids)

    def _pool_subwords(self, sentence_embedding, sentence_word_ids) -> list[torch.Tensor]:
        """Pool sub-word embeddings into word embeddings. Methods: 'first', 'last', 'mean'."""
        word_embeddings: list[torch.Tensor] = []
        subword_embeddings: list[torch.Tensor] = []
        previous_word_id: int = 0

        # Gather word-level embeddings as lists of sub-words
        for token_embedding, word_id in zip(sentence_embedding, sentence_word_ids):
            if previous_word_id != word_id and subword_embeddings:
                word_embeddings.append(torch.stack(subword_embeddings, dim=0))
                subword_embeddings = []

            if word_id is not None:
                subword_embeddings.append(token_embedding)
                previous_word_id = word_id

        # Add last word, some tokenizers don't have 'end of sequence' token
        if subword_embeddings:
            word_embeddings.append(torch.stack(subword_embeddings, dim=0))

        if self.subword_pooling == "first":
            word_embeddings = [word_embedding[0] for word_embedding in word_embeddings]

        if self.subword_pooling == "last":
            word_embeddings = [word_embedding[-1] for word_embedding in word_embeddings]

        if self.subword_pooling == "mean":
            word_embeddings = [torch.mean(word_embedding, dim=0) for word_embedding in word_embeddings]

        word_embeddings = torch.stack(word_embeddings, dim=0)
        return word_embeddings

    def _pool_words(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Pool word embeddings into a text embedding. Methods: 'first', 'last', 'mean', 'weighted_mean'."""
        sentence_embedding = torch.zeros_like(word_embeddings[0])

        # Use first word: for models that use CLS in pre-training
        if self.sentence_pooling == "first":
            sentence_embedding = word_embeddings[0]

        # Mean all words: generally ok for all types of models
        if self.sentence_pooling == "mean":
            sentence_embedding = torch.mean(word_embeddings, dim=0)

        # Use last word: for causal LMs like gpt
        if self.sentence_pooling == "last":
            sentence_embedding = word_embeddings[-1]

        # Less weight on last word: can be better for causal LMs
        if self.sentence_pooling == "weighted_mean":
            weights = torch.linspace(0.9, 0.1, steps=len(word_embeddings))
            weights /= weights.sum()
            sentence_embedding = (word_embeddings * weights[:, None, None]).sum(dim=0)

        return sentence_embedding
