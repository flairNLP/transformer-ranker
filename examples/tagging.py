from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load the WNUT-17 dataset (tweets with named entity tags)
dataset = load_dataset("leondz/wnut_17")

# Create your own list of models
language_models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize the ranker and set the label column
ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2, label_column="ner_tags")

# ... run it with lms
result = ranker.run(models=language_models, batch_size=64)

print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: 2.8448
Rank 2. bert-base-uncased: 2.653
Rank 3. google/electra-small-discriminator: 1.3075
Rank 4. prajjwal1/bert-tiny: 0.7549
"""
