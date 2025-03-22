from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load the WNUT-17 NER dataset of English tweets
dataset_ner = load_dataset("leondz/wnut_17")

# Use smaller models to test on CPU
models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize the ranker, set labels to ner tags
ranker = TransformerRanker(dataset=dataset_ner, dataset_downsample=0.2, label_column="ner_tags")

# ... and run it
result = ranker.run(models=models, batch_size=64)

# Print the scores
print(result)

"""Result
Rank 1. microsoft/deberta-v3-small: 2.8448
Rank 2. bert-base-uncased: 2.653
Rank 3. google/electra-small-discriminator: 1.3075
Rank 4. prajjwal1/bert-tiny: 0.7549
"""
