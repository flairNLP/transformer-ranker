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
    """Test for output types"""
    for model in small_language_models:
        embedder = Embedder(model=model)

        for sentence in test_sentences:
            embeddings = embedder.embed(sentence)
            assert isinstance(embeddings, list)
            assert all(isinstance(emb, torch.Tensor) for emb in embeddings)
            assert all(emb.dim() > 0 for emb in embeddings)


def test_embedder_word_level(small_language_models):
    """Test dimensions for word embeddings"""
    for model in small_language_models:
        embedder = Embedder(model=model)
        embedding = embedder.embed("this is a test sentence")  # 5 words

        # Embedding dim should be (5 words x num_layers x hidden_size)
        assert embedding[0].shape == (5, embedder.num_layers, embedder.hidden_size)


def test_embedder_sentence_level(small_language_models):
    """Test dimensions for text embeddings"""
    for model in small_language_models:
        embedder = Embedder(model=model, sentence_pooling="mean")
        embedding = embedder.embed("this is a test sentence")

        # Embedding dim should be (num_layers x hidden_size)
        assert embedding[0].shape == (embedder.num_layers, embedder.hidden_size)


def test_embedder_layermean(small_language_models):
    """Test dimensions when layer mean is set"""
    for model in small_language_models:
        embedder = Embedder(model=model, layer_mean=True)
        embedding = embedder.embed("this is a test sentence")

        # Averaged hidden states should be (5 words x 1 average x hidden)
        assert embedding[0].shape == (5, 1, embedder.hidden_size)


def test_embedder_layer_selection(small_language_models):
    _, electra_small = small_language_models
    embedder = Embedder(electra_small)

    layer_ids = embedder._parse_layer_ids("all")
    assert layer_ids == list(range(embedder.num_layers))

    layer_ids = embedder._parse_layer_ids("0,1,2,3")  # positive ids
    assert layer_ids == [0, 1, 2, 3]

    layer_ids = embedder._parse_layer_ids("-1,-2,-3, -4")  # negative ids
    assert layer_ids == [8, 9, 10, 11]

    layer_ids = embedder._parse_layer_ids("0,-1,2,-3")  # mixed
    assert layer_ids == [0, 2, 9, 11]

    with pytest.raises(ValueError):
        embedder._parse_layer_ids("100")  # out-of-range


def test_embedder_edge_cases(small_language_models):
    """Test embedder with unusual cases"""
    for model in small_language_models:
        embedder = Embedder(model=model)

        edge_cases = [
            "",
            [],
            "🍕",
            # "\n",
            # "\n \t",
            "1234567890",
            " ".join(["word"] * 10000),
            "@#$%^&*()!",
        ]

        for text in edge_cases:
            embeddings = embedder.embed(text)
            assert isinstance(embeddings, (list, torch.Tensor))
