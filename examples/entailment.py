# Transformer Ranker - Text Pair Classification

# Step 1: Load a text pair dataset
from datasets import load_dataset

# Load the 'SetFit/rte' dataset used for the Recognizing Textual Entailment (RTE) task
dataset = load_dataset('SetFit/rte')

# Specify all columns needed for entailment-type tasks
text_column, text_pair_column, label_column = "text1", "text2", "label"

# Step 2: Prepare a list of transformer handles
models = ['prajjwal1/bert-tiny',
          'google/electra-small-discriminator',
          'microsoft/deberta-v3-small',
          'bert-base-uncased',
          ]

# Step 3: Initialize the ranker and run it
from transformer_ranker import TransformerRanker

ranker = TransformerRanker(dataset=dataset,
                           text_column=text_column,
                           text_pair_column=text_pair_column,  # add text pair column for entailment-type tasks
                           label_column=label_column,
                           )

result = ranker.run(models=models, batch_size=32)

# Step 4: Review Ranked Models
print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: 0.3741
Rank 2. bert-base-uncased: 0.3012
Rank 3. google/electra-small-discriminator: 0.1531
Rank 4. prajjwal1/bert-tiny: 0.0575
"""
