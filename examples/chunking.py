from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load CoNLL-03 dataset
dataset = load_dataset("conll2003")

# Create your own list of models
language_models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize the ranker and set the label column
ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2, label_column="chunk_tags")

# ... run it with language models
result = ranker.run(models=language_models, batch_size=64)

print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: 4.398
Rank 2. bert-base-uncased: 4.149
Rank 3. google/electra-small-discriminator: 3.7423
Rank 4. prajjwal1/bert-tiny: 2.9444
"""
