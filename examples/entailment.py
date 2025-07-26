from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load RTE
dataset = load_dataset("glue", "rte")

# Define models to rank
language_models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize ranker with text pair column
ranker = TransformerRanker(dataset=dataset, text_pair_column="sentence2")

# Run ranking
result = ranker.run(language_models, batch_size=32)
print(result)
