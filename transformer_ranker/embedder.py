from transformers import AutoModel, AutoTokenizer
from tokenizers.pre_tokenizers import Whitespace
import torch

from tqdm import tqdm
from typing import Optional, List, Union


class Embedder:
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        tokenizer: Union[str, AutoTokenizer] = None,
        layer_ids: str = "all",
        subword_pooling: str = "mean",
        layer_pooling: Optional[str] = None,
        sentence_pooling: Optional[str] = None,
        use_pretokenizer: bool = True,
        device: Optional[str] = None,
    ):
        """
        Embedd sentences using any pre-trained transformer model. It's a word-level embedder, meaning that a sentence is
        represented by a list of word vectors. If specified, words can be pooled into a sentence-level embedding.
        ♻️ Feel free to use it if you ever need a simple implementation for word-level embeddings.

        :param model: Name of the model to be used. Either a model handle (e.g. 'bert-base-uncased')
        or a loaded model e.g. AutoModel('bert-base-uncased').
        :param tokenizer: Optional parameter to specify the tokenizer. Either a tokenizer handle
        (e.g. 'bert-base-uncased') or a loaded tokenizer e.g. AutoTokenizer.from_pretrained('bert-base-uncased').
        :param subword_pooling: Method used to pool sub-word embeddings to form word-level embeddings.
        :param layer_ids: Specifies which layers' outputs should be used. This can be a single top-most layer as '-1',
        multiple layers like '-1,-2,-3, -4', or 'all' to use all layers. Default is 'all'.
        :param layer_pooling: Optional method used to combine or pool embeddings from selected layers.
        If not specified, no pooling across layers is applied, and each layer's output is handled independently.
        :param use_pretokenizer: If to pre-tokenize texts using whitespace
        :param device: Optional specification of the computing device where the model operations are performed.
        Can be 'cpu' or 'cuda'. If not specified, it defaults to the best available device.
        """
        # Load transformer model
        self.model = model if isinstance(model, torch.nn.Module) else AutoModel.from_pretrained(model)
        self.model_name = model if isinstance(model, str) else self.model.config.name_or_path

        # Load a model-specific tokenizer
        self.tokenizer = (tokenizer if tokenizer
                          else AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=True))

        # Add padding token for models that do not have it (e.g. GPT2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Use whitespace pre-tokenizer if specified
        self.pre_tokenizer = Whitespace() if use_pretokenizer else None

        # Get number of layers from config
        self.num_transformer_layers = self.model.config.num_hidden_layers

        # Set relevant layers that will be used for embeddings
        self.layer_ids = self._filter_layer_ids(layer_ids)

        # Set pooling operations for sub-words and layers
        self.subword_pooling = subword_pooling
        self.layer_pooling = layer_pooling

        # Set sentence-pooling to get embedding for the full text if specified
        self.sentence_pooling = sentence_pooling

        # Set cpu or gpu device
        self.device = (device if device else
                       torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device)

    def tokenize(self, sentences):
        """Tokenize sentences using auto tokenizer"""
        # Handle tokenizers with wrong model_max_length in hf configuration
        max_sequence_length = self.tokenizer.model_max_length if self.tokenizer.model_max_length < 1000000 else 512

        # Pre-tokenize sentences using hf whitespace tokenizer
        if self.pre_tokenizer and isinstance(sentences[0], str):
            sentences = [[word for word, word_offsets in self.pre_tokenizer.pre_tokenize_str(sentence)]
                         for sentence in sentences]

        is_split_into_words = False if isinstance(sentences[0], str) else True

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
        move_embeddings_to_cpu: bool = True
    ) -> List[torch.Tensor]:
        """Split sentences into batches and embedd the full dataset"""
        if not isinstance(sentences, list):
            sentences = [sentences]

        batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        embeddings = []
        tqdm_bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

        for batch in tqdm(
            batches,
            desc="Retrieving Embeddings ",
            disable=not show_loading_bar,
            bar_format=tqdm_bar_format
        ):
            embeddings.extend(self.embed_batch(batch, move_embeddings_to_cpu))

        return embeddings

    def embed_batch(self, sentences, move_embeddings_to_cpu: bool = True) -> List[torch.Tensor]:
        """Embeds a batch of sentences and returns a list of sentence embeddings that
        consist of different numbers of word-level embeddings"""
        # Tokenize with auto tokenizer
        tokenized_input = self.tokenize(sentences)

        # Move inputs to gpu
        input_ids = tokenized_input["input_ids"].to(self.device)
        attention_mask = tokenized_input["attention_mask"].to(self.device)
        word_ids = [tokenized_input.word_ids(i) for i in range(len(sentences))]

        # Embedd using a transformer: forward pass to get all hidden states of the model
        with torch.no_grad():
            hidden_states = self.model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            ).hidden_states

        # Exclude the embedding layer (index 0)
        embeddings = torch.stack(hidden_states[1:], dim=0)
        embeddings = embeddings.permute(1, 2, 0, 3)

        # Multiply embeddings by attention mask to have padded tokens as 0
        embeddings = embeddings * attention_mask.unsqueeze(-1).unsqueeze(-1)

        # Extract layers defined by layer_ids, average all layers for a batch of sentences if specified
        embeddings = self._extract_relevant_layers(embeddings)

        # Process each sentence separately and gather word or sentence embeddings
        sentence_embeddings = []
        for subword_embeddings, word_ids in zip(embeddings, word_ids):

            # Pool sub-words to get word-level embeddings
            word_embeddings = self._pool_subwords(subword_embeddings, word_ids)

            # Stack all word-level embeddings that represent a sentence
            word_embeddings = torch.stack(word_embeddings, dim=0)

            # Pool word-level embeddings into a single sentence vector if specified
            sentence_embedding = self._pool_words(word_embeddings) if self.sentence_pooling else word_embeddings

            # Store sentence-embedding tensors in a python list
            sentence_embeddings.append(sentence_embedding)

        # Move batch of embeddings to cpu
        if move_embeddings_to_cpu:
            sentence_embeddings = [sentence_embedding.cpu() for sentence_embedding in sentence_embeddings]

        return sentence_embeddings

    def _filter_layer_ids(self, layer_ids):
        """Transform a string with layer ids into a list of ints and
         remove ids that are out of bound of the actual transformer size"""
        if layer_ids == "all":
            layer_ids = ", ".join([str(-1 * (i + 1)) for i in range(self.num_transformer_layers)])

        layer_ids = [int(number) for number in layer_ids.split(",")]

        layer_ids = [layer_id for layer_id in layer_ids if self.num_transformer_layers + 1 >= abs(layer_id)]
        return layer_ids

    def _extract_relevant_layers(self, batched_embeddings: torch.Tensor) -> torch.Tensor:
        """Keep only relevant layers in each embedding and apply layer-wise pooling if required"""
        # To maintain original transformer layer order, map negative layer IDs to corresponding positive indices,
        layer_ids = sorted((layer_id if layer_id >= 0 else self.num_transformer_layers + layer_id)
                           for layer_id in self.layer_ids)

        # A batch of raw embeddings is in this shape (batch_size, sequence_length, num_layers - 1, hidden_size)
        batched_embeddings = batched_embeddings[:, :, layer_ids, :]  # keep only selected layers

        # Apply mean pooling over the layer dimension if specified
        if self.layer_pooling == "mean":
            batched_embeddings = torch.mean(batched_embeddings, dim=2, keepdim=True)

        return batched_embeddings

    def _pool_subwords(self, sentence_embedding, sentence_word_ids) -> List[torch.Tensor]:
        """Pool sub-word embeddings into word embeddings for a single sentence.
        Subword pooling methods: 'first', 'last', 'mean'"""
        word_embeddings = []
        subword_embeddings = []
        previous_word_id = 0

        # Gather word-level embeddings as lists of subwords
        for token_embedding, word_id in zip(sentence_embedding, sentence_word_ids):

            # Append a word (stack all subwords into a word tensor)
            if previous_word_id != word_id and subword_embeddings:
                word_embeddings.append(torch.stack(subword_embeddings, dim=0))
                subword_embeddings = []

            # Gather subword tokens into a single word
            if word_id is not None:
                subword_embeddings.append(token_embedding)
                previous_word_id = word_id

        # Append the last word (some tokenizers don't have 'end of sequence' token)
        if subword_embeddings:
            word_embeddings.append(torch.stack(subword_embeddings, dim=0))

        if self.subword_pooling == "first":
            word_embeddings = [word_embedding[0] for word_embedding in word_embeddings]

        if self.subword_pooling == "last":
            word_embeddings = [word_embedding[-1] for word_embedding in word_embeddings]

        if self.subword_pooling == "mean":
            word_embeddings = [torch.mean(word_embedding, dim=0) for word_embedding in word_embeddings]

        return word_embeddings

    def _pool_words(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Pool word embeddings into a sentence embedding.
        Subword pooling methods: 'first', 'last', 'mean', 'weighted_mean'"""
        sentence_embedding = torch.zeros_like(word_embeddings[0])

        # Use the first word as a sentence embedding: for models that use CLS token in pre-training
        if self.sentence_pooling == "first":
            sentence_embedding = word_embeddings[0]

        # Mean all word-level embeddings: generally ok for all types of models
        if self.sentence_pooling == "mean":
            sentence_embedding = torch.mean(word_embeddings, dim=0)

        # Use the last word as a sentence embedding: for Causal LMs (autoregressive)
        if self.sentence_pooling == "last":
            sentence_embedding = word_embeddings[-1]

        # Weight words by last word having the lowest importance: slightly better option for Causal LMs
        if self.sentence_pooling == "weighted_mean":
            weights = torch.linspace(0.9, 0.1, steps=len(word_embeddings))
            weights /= weights.sum()
            sentence_embedding = (word_embeddings * weights[:, None, None]).sum(dim=0)

        return sentence_embedding
