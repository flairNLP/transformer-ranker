from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load STS-B (Regression)
dataset = load_dataset("glue", "stsb")

# Define models to rank
language_models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize ranker for text-pair regression
ranker = TransformerRanker(dataset=dataset, text_pair_column="sentence2")

# Run ranking with LogME
result = ranker.run(language_models, estimator="logme")
print(result)
