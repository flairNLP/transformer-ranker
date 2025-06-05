from datasets import load_dataset
from transformer_ranker import TransformerRanker

dataset = load_dataset("trec")

ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2)

# Run 1: Rank small models
small_models = ["prajjwal1/bert-tiny", "google/electra-small-discriminator"]

# ... using a large batch size
result = ranker.run(models=small_models, batch_size=128)

# Run 2: Add rankings of larger models
large_models = ["bert-large-cased", "google/electra-large-discriminator"]

## ... using a small batch size
result.append(ranker.run(batch_size=16, models=large_models))

# Look at merged results
print(result)

"""Result:
Rank 1. google/electra-large-discriminator: 4.2713
Rank 2. bert-large-cased: 4.1123
Rank 3. google/electra-small-discriminator: 2.9056
Rank 4. prajjwal1/bert-tiny: 1.956
"""
