from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load dataset and init the ranker
dataset = load_dataset("trec")
ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2)

# First run: small models with large batch size
small_models = ["prajjwal1/bert-tiny", "google/electra-small-discriminator"]
result = ranker.run(models=small_models, batch_size=128)

# Second run: large models with small batch size
large_models = ["bert-large-cased", "google/electra-large-discriminator"]
result.append(ranker.run(models=large_models, batch_size=16))

# Combined results
print(result)