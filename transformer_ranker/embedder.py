from typing import List, Optional, Union

import torch
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast


class Embedder:
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        tokenizer: Union[str, PreTrainedTokenizerFast, None] = None,
        layer_ids: str = "all",
        subword_pooling: str = "mean",
        layer_pooling: Optional[str] = None,
        sentence_pooling: Optional[str] = None,
        use_pretokenizer: bool = True,
        local_files_only: bool = False,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Embed texts using a pre-trained transformer model. It's a word-level embedder, where
        each text is a list of word embeddings. Supports sub-word and sentence pooling options.
        ♻️ Feel free to use it if you need a simple implementation for word or text embeddings.

        :param model: Model name 'bert-base-uncased' or a model instance loaded with AutoModel
        :param tokenizer: Optional tokenizer, either a string name or a tokenizer instance.
        :param subword_pooling: Method for pooling sub-words into word embeddings.
        :param layer_ids: Layers to use e.g., '-1' for the top-most layer or 'all'.
        :param layer_pooling: Optional method for pooling across selected layers.
        :param use_pretokenizer: Whether to pre-tokenize texts using whitespace.
        :param device: Device option, either 'cpu' or 'cuda:0'. Defaults to the available device.
        """
        # Load transformer model
        if isinstance(model, torch.nn.Module):
            self.model = model
            self.model_name = model.config.name_or_path
        else:
            self.model = AutoModel.from_pretrained(model, local_files_only=local_files_only)
            self.model_name = model

        # Load a model-specific tokenizer
        self.tokenizer: PreTrainedTokenizerFast
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            self.tokenizer = tokenizer
        else:
            tokenizer_source = tokenizer if isinstance(tokenizer, str) else self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                add_prefix_space=True,
                clean_up_tokenization_spaces=True,
            )

        # Add padding token for models that do not have it (e.g. GPT2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Use whitespace pre-tokenizer if specified
        self.pre_tokenizer = Whitespace() if use_pretokenizer else None

        # Get number of layers from config
        self.num_transformer_layers = self.model.config.num_hidden_layers

        # Set relevant layers that will be used for embeddings
        self.layer_ids = self._filter_layer_ids(layer_ids)

        # Set pooling options
        self.subword_pooling = subword_pooling
        self.layer_pooling = layer_pooling
        self.sentence_pooling = sentence_pooling

        # Move model to device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device)

    def tokenize(self, sentences):
        """Tokenize sentences using auto tokenizer"""
        # Handle tokenizers with wrong model_max_length in hf configuration
        max_sequence_length = (
            self.tokenizer.model_max_length if self.tokenizer.model_max_length < 1000000 else 512
        )

        # Pre-tokenize sentences using hf whitespace tokenizer
        if self.pre_tokenizer and isinstance(sentences[0], str):
            sentences = [
                [word for word, word_offsets in self.pre_tokenizer.pre_tokenize_str(sentence)]
                for sentence in sentences
            ]

        is_split_into_words = not isinstance(sentences[0], str)

        # Use model-specific tokenizer and return output as tensors
        return self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
            is_split_into_words=is_split_into_words,
        )

    def embed(
        self,
        sentences,
        batch_size: int = 32,
        show_loading_bar: bool = True,
        move_embeddings_to_cpu: bool = True,
    ) -> List[torch.Tensor]:
        """Split sentences into batches and embedd the full dataset"""
        if not isinstance(sentences, list):
            sentences = [sentences]

        batches = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]
        embeddings = []
        tqdm_bar_format = "{l_bar}{bar:10}{r_bar}{bar:-10b}"

        for batch in tqdm(
            batches,
            desc="Retrieving Embeddings",
            disable=not show_loading_bar,
            bar_format=tqdm_bar_format,
        ):
            embeddings.extend(self.embed_batch(batch, move_embeddings_to_cpu))

        return embeddings

    def embed_batch(self, sentences, move_embeddings_to_cpu: bool = True) -> List[torch.Tensor]:
        """Embeds a batch of sentences and returns a list of sentence embeddings
        (list of word embeddings). Embeddings can be moved to cpu or kept on gpu"""
        tokenized_input = self.tokenize(sentences)

        # Move inputs to gpu
        input_ids = tokenized_input["input_ids"].to(self.device)
        attention_mask = tokenized_input["attention_mask"].to(self.device)
        word_ids = [tokenized_input.word_ids(i) for i in range(len(sentences))]

        # Embedd: forward pass to get all hidden states of the model
        with torch.no_grad():
            hidden_states = self.model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            ).hidden_states

        # Exclude the embedding layer (index 0)
        embeddings = torch.stack(hidden_states[1:], dim=0)
        embeddings = embeddings.permute(1, 2, 0, 3)

        # Multiply embeddings by attention mask to have padded tokens as 0
        embeddings = embeddings * attention_mask.unsqueeze(-1).unsqueeze(-1)

        # Extract and average specified layers
        embeddings = self._extract_relevant_layers(embeddings)

        # Go through each sentence separately
        sentence_embeddings = []
        for subword_embeddings, word_ids in zip(embeddings, word_ids):

            # Pool sub-words to get word-level embeddings
            word_embedding_list = self._pool_subwords(subword_embeddings, word_ids)

            # Stack all word-level embeddings that represent a sentence
            word_embeddings = torch.stack(word_embedding_list, dim=0)

            # Pool word-level embeddings into a sentence embedding
            sentence_embedding = (
                self._pool_words(word_embeddings) if self.sentence_pooling else word_embeddings
            )

            sentence_embeddings.append(sentence_embedding)

        if move_embeddings_to_cpu:
            sentence_embeddings = [
                sentence_embedding.cpu() for sentence_embedding in sentence_embeddings
            ]

        return sentence_embeddings

    def _filter_layer_ids(self, layer_ids: str) -> List[int]:
        """Transform a string with layer ids into a list of ints.
        Check if any ids are out-of-bounds of model size"""
        num_layers = self.num_transformer_layers
        if layer_ids == "all":
            new_layer_ids = [-i for i in range(1, num_layers + 1)]
        else:
            new_layer_ids = [int(number) for number in layer_ids.split(",")]
            new_layer_ids = [layer_id for layer_id in new_layer_ids if abs(layer_id) <= num_layers]

        if not new_layer_ids:
            raise ValueError(
                f'Given layer_ids are out of bounds for the model size. '
                f"Num layers in model {self.model_name}: {num_layers}"
            )

        return new_layer_ids

    def _extract_relevant_layers(self, batched_embeddings: torch.Tensor) -> torch.Tensor:
        """Keep only relevant layers in each embedding and apply layer-wise pooling if required"""
        # Use positive layer ids ('-1 -> 23' is the last layer in a 24 layer model)
        layer_ids = sorted(
            (layer_id if layer_id >= 0 else self.num_transformer_layers + layer_id)
            for layer_id in self.layer_ids
        )

        # Embeddings shape: (batch_size, seq_len, num_layers, hidden_size)
        batched_embeddings = batched_embeddings[:, :, layer_ids, :]  # keep only selected layers

        # average all layers
        if self.layer_pooling == "mean":
            batched_embeddings = torch.mean(batched_embeddings, dim=2, keepdim=True)

        return batched_embeddings

    def _pool_subwords(self, sentence_embedding, sentence_word_ids) -> List[torch.Tensor]:
        """Pool sub-word embeddings into word embeddings for a single sentence.
        Subword pooling methods: 'first', 'last', 'mean'"""
        word_embeddings: List[torch.Tensor] = []
        subword_embeddings: List[torch.Tensor] = []
        previous_word_id: int = 0

        # Gather word-level embeddings as lists of sub-words
        for token_embedding, word_id in zip(sentence_embedding, sentence_word_ids):

            # Stack all sub-words into a word tensor
            if previous_word_id != word_id and subword_embeddings:
                word_embeddings.append(torch.stack(subword_embeddings, dim=0))
                subword_embeddings = []

            if word_id is not None:
                subword_embeddings.append(token_embedding)
                previous_word_id = word_id

        # Add last word as some tokenizers don't have 'end of sequence' token
        if subword_embeddings:
            word_embeddings.append(torch.stack(subword_embeddings, dim=0))

        if self.subword_pooling == "first":
            word_embeddings = [word_embedding[0] for word_embedding in word_embeddings]

        if self.subword_pooling == "last":
            word_embeddings = [word_embedding[-1] for word_embedding in word_embeddings]

        if self.subword_pooling == "mean":
            word_embeddings = [
                torch.mean(word_embedding, dim=0) for word_embedding in word_embeddings
            ]

        return word_embeddings

    def _pool_words(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Pool word embeddings into a sentence embedding.
        Subword pooling methods: 'first', 'last', 'mean', 'weighted_mean'"""
        sentence_embedding = torch.zeros_like(word_embeddings[0])

        # Use first word as sentence embedding: for models that use CLS in pre-training
        if self.sentence_pooling == "first":
            sentence_embedding = word_embeddings[0]

        # Mean all word-level embeddings: generally ok for all types of models
        if self.sentence_pooling == "mean":
            sentence_embedding = torch.mean(word_embeddings, dim=0)

        # Use the last word as sentence embedding: for Causal LMs (autoregressive)
        if self.sentence_pooling == "last":
            sentence_embedding = word_embeddings[-1]

        # Weight words by last word having lower importance: can be better for Causal LMs
        if self.sentence_pooling == "weighted_mean":
            weights = torch.linspace(0.9, 0.1, steps=len(word_embeddings))
            weights /= weights.sum()
            sentence_embedding = (word_embeddings * weights[:, None, None]).sum(dim=0)

        return sentence_embedding
