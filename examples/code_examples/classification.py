from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load and inspect the 'trec' dataset
dataset = load_dataset('trec')
print(dataset)

# Use smaller models to run on CPU
language_models = ['prajjwal1/bert-tiny',
                   'google/electra-small-discriminator',
                   'microsoft/deberta-v3-small',
                   'bert-base-uncased',
                   ]

# Initialize the ranker
ranker = TransformerRanker(dataset=dataset,
                           dataset_downsample=0.2,
                           label_column="coarse_label",
                           )

# ... and run it
result = ranker.run(models=language_models, batch_size=32)

# Print the scores
print(result)

"""Result: 
Rank 1. microsoft/deberta-v3-small: 4.0819
Rank 2. bert-base-uncased: 3.9312
Rank 3. google/electra-small-discriminator: 2.9627
Rank 4. prajjwal1/bert-tiny: 1.998
"""
