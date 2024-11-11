from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load 'rte' Recognizing Textual Entailment dataset
entailment_dataset = load_dataset('glue', 'rte')

# Use smaller models to run on CPU
language_models = ['prajjwal1/bert-tiny',
                   'google/electra-small-discriminator',
                   'microsoft/deberta-v3-small',
                   'bert-base-uncased',
                   ]

# Initialize the ranker, set text_pair_column
ranker = TransformerRanker(dataset=entailment_dataset, text_pair_column="sentence2")

# ... and run it
result = ranker.run(models=language_models, batch_size=32)

# Print the scores
print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: 0.3741
Rank 2. bert-base-uncased: 0.3012
Rank 3. google/electra-small-discriminator: 0.1531
Rank 4. prajjwal1/bert-tiny: 0.0575
"""
