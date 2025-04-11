from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load STS-B text-pair regression dataset
dataset = load_dataset("glue", "stsb")

# Create your own list of models
language_models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize the ranker and set the text pair column
ranker = TransformerRanker(dataset=dataset, text_pair_column="sentence2")

# ... run it using LogME
result = ranker.run(models=language_models, estimator="logme")

print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: -1.1082
Rank 2. google/electra-small-discriminator: -1.2708
Rank 3. bert-base-uncased: -1.3015
Rank 4. prajjwal1/bert-tiny: -1.7165
"""
