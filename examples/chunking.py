from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load CoNLL-03
dataset = load_dataset("conll2003")

# Define models to rank
language_models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize ranker
ranker = TransformerRanker(
    dataset=dataset,
    dataset_downsample=0.2,
    label_column="chunk_tags"
)

# Run ranking
result = ranker.run(language_models, batch_size=64)
print(result)