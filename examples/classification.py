from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load TREC
dataset = load_dataset("trec")

# Define models to rank
language_models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize ranker
ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2)

# Run ranking
result = ranker.run(language_models, batch_size=32)
print(result)
