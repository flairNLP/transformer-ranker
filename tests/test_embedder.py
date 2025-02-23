import pytest
import torch
from transformer_ranker import Embedder


test_sentences = [
    "this is a test sentence",
    ["this", "is", "a", "test", "sentence"],
    ["this is the first sentence.", "this is the second sentence."],
    [["this", "is", "the", "first", "sentence", "."], ["this", "is", "the", "second", "sentence", "."]],
]


def test_embedder_outputs(small_language_models):
    """Tests embedder outputs with sample sentences."""
    for model in small_language_models:
        embedder = Embedder(model=model)

        for sentence in test_sentences:
            embeddings = embedder.embed(sentence)
            assert isinstance(embeddings, list), f"Expected list, but found {type(embeddings)}"
            assert isinstance(embeddings, list) and all(isinstance(emb, torch.Tensor) for emb in embeddings), \
                "Expected list of tensors, but got different structure or type."
            assert all(emb.dim() > 0 for emb in embeddings), "Each tensor should have at least one dim."


def test_embedder_word_level(small_language_models):
    """Test word embeddings"""
    for model in small_language_models:
        embedder = Embedder(model=model)
        model_name = embedder.name
        embedding = embedder.embed("this is a test sentence")  # 5 words

        # Embedding dim should be (5 words x num_layers x hidden_size)
        assert embedding[0].shape == (5, embedder.num_layers, embedder.hidden_size), \
            f"bad shape (5,{embedder.num_layers},{embedder.hidden_size}) != {embedding[0].shape} for {model_name}"


def test_embedder_sentence_level(small_language_models):
    """Test embedder with sentence pooling"""
    for model in small_language_models:
        embedder = Embedder(model=model, sentence_pooling="mean")
        model_name = embedder.name
        embedding = embedder.embed("this is a test sentence")

        # Embedding dim should be (num_layers x hidden_size)
        assert embedding[0].shape == (embedder.num_layers, embedder.hidden_size), \
            f"bad shape (1, {embedder.hidden_size}) != {embedding[0].shape} for {model_name}"


def test_embedder_layermean(small_language_models):
    """Test embedding with hidden state average"""
    for model in small_language_models:
        embedder = Embedder(model=model, layer_mean=True)
        model_name = embedder.name
        embedding = embedder.embed("this is a test sentence")

        # Averaged hidden states should be (5 words x 1 average x hidden)
        assert embedding[0].shape == (5, 1, embedder.hidden_size), \
            f"bad shape (5,1,{embedder.hidden_size}) != {embedding[0].shape} for {model_name}"


def test_embedder_layers(small_language_models):
    """Test embedder layer ids param"""
    _, electra_small = small_language_models
    with pytest.raises(ValueError):
        Embedder(model=electra_small, layer_ids="13")  # layer 13 should be out-of-bounds

    embedder = Embedder(model=electra_small, layer_ids="all")
    embedding = embedder.embed("word")[0]
    assert (
        embedding.shape[1] == embedder.num_layers
    ), f"Expected to have an embedding for all {embedder.num_layers} layers, but got {embedding.shape}"

    embedder = Embedder(model=electra_small, layer_ids="0")
    embedding = embedder.embed("word")[0]
    assert embedding.shape[1] == 1, f"Expected to have an embedding for a single layer, but got {embedding.shape}"


def test_embedder_edge_cases(small_language_models):
    """Test embedder with edgy sentences"""
    for model in small_language_models:
        embedder = Embedder(model=model)

        edge_cases = [
            "",
            [],
            " ".join(["word"] * 10000),
            "@#$%^&*()!",
        ]

        for text in edge_cases:
            embeddings = embedder.embed(text)
            assert isinstance(embeddings, (list, torch.Tensor)), f"Unexpected output for {text}: {type(embeddings)}"

