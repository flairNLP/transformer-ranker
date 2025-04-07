import pytest

from transformer_ranker import TransformerRanker


def test_ranker_sick(small_language_models, sick):
    """Test ranker on the SICK, text pair classification dataset."""
    ranker = TransformerRanker(dataset=sick, text_pair_column="text2", dataset_downsample=0.025)
    result = ranker.run(small_language_models, batch_size=64)
    assert len(str(result).split("\n")) >= 2


def test_ranker_conll(small_language_models, conll):
    """Test ranker on the CoNLL, NER dataset."""
    ranker = TransformerRanker(dataset=conll, dataset_downsample=0.01)
    result = ranker.run(small_language_models, batch_size=64)
    assert len(str(result).split("\n")) >= 2


def test_ranker_wnut(small_language_models, wnut):
    """Test ranker on the WNUT, NER dataset."""
    ranker = TransformerRanker(dataset=wnut, dataset_downsample=0.05)
    result = ranker.run(small_language_models, batch_size=64)
    assert len(str(result).split("\n")) >= 2


def test_ranker_trec(small_language_models, trec):
    """Test ranker on the TREC, text classification dataset."""
    ranker = TransformerRanker(dataset=trec, dataset_downsample=0.05)
    result = ranker.run(small_language_models, batch_size=64)
    assert len(str(result).split("\n")) >= 2


def test_ranker_stsb(small_language_models, stsb):
    """Test ranker on the STS-B, text pair regression dataset."""
    ranker = TransformerRanker(dataset=stsb, dataset_downsample=0.05)

    with pytest.raises(ValueError, match=r"HScore is not suitable for regression"):
        ranker.run(small_language_models, estimator="hscore", batch_size=64)

    result = ranker.run(small_language_models, estimator="logme", batch_size=64)
    assert len(str(result).split("\n")) >= 2


def test_ranker_bestlayer(small_language_models, trec):
    """Test ranker with 'bestlayer' aggregation."""
    ranker = TransformerRanker(dataset=trec, dataset_downsample=0.05)
    result = ranker.run(small_language_models, layer_aggregator='bestlayer')
    assert len(str(result).split("\n")) >= 2
