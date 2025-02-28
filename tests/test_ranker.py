import pytest
from transformer_ranker import TransformerRanker


def test_ranker_sick(small_language_models, sick):
    ranker = TransformerRanker(dataset=sick, text_pair_column='text2', dataset_downsample=0.025)
    result = ranker.run(small_language_models, batch_size=64)
    assert len(str(result).split('\n')) >= 2


def test_ranker_conll(small_language_models, conll):
    ranker = TransformerRanker(dataset=conll, dataset_downsample=0.01)
    result = ranker.run(small_language_models, batch_size=64)
    assert len(str(result).split('\n')) >= 2


def test_ranker_wnut(small_language_models, wnut):
    ranker = TransformerRanker(dataset=wnut, dataset_downsample=0.05)
    result = ranker.run(small_language_models, batch_size=64)
    assert len(str(result).split('\n')) >= 2


def test_ranker_trec(small_language_models, trec):
    ranker = TransformerRanker(dataset=trec, dataset_downsample=0.05)
    result = ranker.run(small_language_models, batch_size=64)
    assert len(str(result).split('\n')) >= 2


def test_ranker_stsb(small_language_models, stsb):
    ranker = TransformerRanker(dataset=stsb, dataset_downsample=0.05)

    with pytest.raises(ValueError, match=r"HScore is not suitable for regression"):
        ranker.run(small_language_models, estimator='hscore', batch_size=64)

    result = ranker.run(small_language_models, estimator='logme', batch_size=64)
    assert len(str(result).split('\n')) >= 2


def test_ranker_bestlayer(small_language_models, trec):
    ranker = TransformerRanker(dataset=trec, dataset_downsample=0.05)
    result = ranker.run(small_language_models, batch_size=64)
    assert len(str(result).split('\n')) >= 2
