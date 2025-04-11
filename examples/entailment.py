from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load RTE (Recognizing Textual Entailment) dataset
dataset = load_dataset("glue", "rte")

# Create your own list of models
language_models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize the ranker and set the text pair column
ranker = TransformerRanker(dataset=dataset, text_pair_column="sentence2")

# ... run it
result = ranker.run(models=language_models, batch_size=32)

print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: 0.3741
Rank 2. bert-base-uncased: 0.3012
Rank 3. google/electra-small-discriminator: 0.1531
Rank 4. prajjwal1/bert-tiny: 0.0575
"""
