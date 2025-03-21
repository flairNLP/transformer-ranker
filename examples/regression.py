from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load a regression dataset
regression_dataset = load_dataset("glue", "stsb")

# Use smaller models to run on CPU
models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize the ranker, set the text pair column
ranker = TransformerRanker(dataset=regression_dataset, text_pair_column="sentence2")

# set transferability estimation to logme for regression tasks
result = ranker.run(models=models, estimator="logme")

# Print the scores
print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: -1.1082
Rank 2. google/electra-small-discriminator: -1.2708
Rank 3. bert-base-uncased: -1.3015
Rank 4. prajjwal1/bert-tiny: -1.7165
"""
