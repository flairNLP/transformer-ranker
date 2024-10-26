import pytest
from transformer_ranker import Embedder


test_sentences = [
    "this is a test sentence",
    ["this", "is", "a", "test", "sentence"],
    ["this is the first sentence.", "this is the second sentence."],
    [["this", "is", "the", "first", "sentence", "."], ["this", "is", "the", "second", "sentence", "."]]
]


def test_embedder_inputs(small_language_models):
    embeddings = {
        'prajjwal1/bert-tiny': [],
        'google/electra-small-discriminator': []
    }

    for model in small_language_models:
        embedder = Embedder(model=model, layer_ids="all")
        model_name = embedder.model_name

        for sentence in test_sentences:
            embedding = embedder.embed(sentence)
            embeddings[model_name].append(embedding)

    for model_name, sentence_embeddings in embeddings.items():
        for embedding in sentence_embeddings:
            assert embedding is not None and embedding != [], f"Empty or None embedding found for model {model_name}"


def test_embedder_outputs(small_language_models):
    for model in small_language_models:
        embedder = Embedder(model=model, layer_ids="all")  # test word-level embedder
        model_name = embedder.model_name
        num_layers = embedder.num_transformer_layers
        embedding = embedder.embed("this is a test sentence")[0]  # 5 words

        # Embedding dim should be 5 words x num_layers x hidden_size
        assert embedding.shape[:2] == (5, num_layers), \
            f"Expected first two dimensions to be (5, {num_layers}), got {embedding.shape[:2]} using model {model_name}"

    for model in small_language_models:
        embedder = Embedder(model=model, layer_ids="all", sentence_pooling="mean")  # test sentence-level embedder
        model_name = embedder.model_name
        num_layers = embedder.num_transformer_layers
        embedding = embedder.embed("this is a test sentence.")[0]

        # Embedding dim should be num_layers x hidden_size
        assert embedding.shape[0] == num_layers, \
            (f"Expected to have a single sentence embedding with dim (1, hidden_size)"
             f"but got {embedding.shape} using model {model_name}")
