from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load 'rte' Recognizing Textual Entailment dataset
dataset = load_dataset('SetFit/rte')

# Add all columns names for entailment-type tasks
text_column, text_pair_column, label_column = "text1", "text2", "label"

# Use smaller models to test on cpu
language_models = ['prajjwal1/bert-tiny',
                   'google/electra-small-discriminator',
                   'microsoft/deberta-v3-small',
                   'bert-base-uncased',
                   ]

# Initialize the ranker
ranker = TransformerRanker(dataset=dataset,
                           text_column=text_column,
                           text_pair_column=text_pair_column,  # let it know where the second column is
                           label_column=label_column,
                           )

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
