import pytest
from transformer_ranker import Embedder


test_sentences = [
    "this is a test sentence",
    ["this", "is", "a", "test", "sentence"],
    ["this is the first sentence.", "this is the second sentence."],
    [["this", "is", "the", "first", "sentence", "."], ["this", "is", "the", "second", "sentence", "."]],
]


def test_embedder_inputs(small_language_models):
    for model in small_language_models:
        embedder = Embedder(model=model, layer_ids="all")

        for sentence in test_sentences:
            embedding = embedder.embed(sentence)
            assert (
                embedding is not None and embedding != []
            ), f"Empty or None embedding found for model {embedder.name}"


def test_embedder_word_level(small_language_models):
    for model in small_language_models:
        embedder = Embedder(model=model, layer_ids="all")
        model_name = embedder.name
        embedding = embedder.embed("this is a test sentence")[0]  # 5 words

        # Embedding dim should be 5 words x num_layers x hidden_size
        assert embedding.shape[:2] == (
            5,
            embedder.num_layers,
        ), f"Shape mismatch: expected (5,{embedder.num_layers}), got {embedding.shape[:2]} for {model_name}"


def test_embedder_sentence_level(small_language_models):
    for model in small_language_models:
        embedder = Embedder(model=model, layer_ids="all", sentence_pooling="mean")
        model_name = embedder.name
        embedding = embedder.embed("this is a test sentence.")[0]

        # Embedding dim should be num_layers x hidden_size
        assert embedding.shape[0] == embedder.num_layers, (
            f"Expected to have a single sentence embedding with dim (1, hidden_size)"
            f"but got {embedding.shape} using model {model_name}"
        )


def test_embedder_layers(small_language_models):
    _, electra_small = small_language_models
    with pytest.raises(ValueError):
        # layer 13 should be out-of-bounds
        Embedder(model=electra_small, layer_ids="13")

    embedder = Embedder(model=electra_small, layer_ids="all")
    embedding = embedder.embed("word")[0]
    assert (
        embedding.shape[1] == embedder.num_layers
    ), f"Expected to have an embedding for all {embedder.num_layers} layers, but got {embedding.shape}"

    embedder = Embedder(model=electra_small, layer_ids="0")
    embedding = embedder.embed("word")[0]
    assert embedding.shape[1] == 1, f"Expected to have an embedding for a single layer, but got {embedding.shape}"
