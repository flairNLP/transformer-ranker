from transformer_ranker import TransformerRanker


def test_ranker_trec(small_language_models, trec):
    ranker = TransformerRanker(dataset=trec, dataset_downsample=0.05)
    ranker.run(models=small_language_models, batch_size=64)


def test_ranker_conll(small_language_models, conll):
    ranker = TransformerRanker(dataset=conll, dataset_downsample=0.01)
    ranker.run(models=small_language_models, batch_size=64)
