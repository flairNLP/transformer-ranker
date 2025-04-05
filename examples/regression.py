from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load STS-B text pair regression dataset
regression_dataset = load_dataset("glue", "stsb")

# Prepare smaller models to run on CPU
models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize the ranker, set the text pair column
ranker = TransformerRanker(dataset=regression_dataset, text_pair_column="sentence2")

# ... run it using LogME for regression
result = ranker.run(models=models, estimator="logme")

print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: -1.1082
Rank 2. google/electra-small-discriminator: -1.2708
Rank 3. bert-base-uncased: -1.3015
Rank 4. prajjwal1/bert-tiny: -1.7165
"""
