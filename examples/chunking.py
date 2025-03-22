from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load the 'conll2003' dataset
dataset = load_dataset("conll2003")

# Use smaller models to run on CPU
models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize the ranker, set labels to chunk tags
ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2, label_column="chunk_tags")

# ... and run it
result = ranker.run(models=models, batch_size=64)

# Print the scores
print(result)

"""Result
Rank 1. microsoft/deberta-v3-small: 4.398
Rank 2. bert-base-uncased: 4.149
Rank 3. google/electra-small-discriminator: 3.7423
Rank 4. prajjwal1/bert-tiny: 2.9444
"""
