# Transformer Ranker - Named Entity Recognition

# Step 1: Load a NER dataset using transformer datasets
from datasets import load_dataset

dataset = load_dataset('conll2003')

# Step 2: Prepare a list of transformer handles
models = ['prajjwal1/bert-tiny',
          'google/electra-small-discriminator',
          'microsoft/deberta-v3-small',
          'bert-base-uncased',
          ]

# Step 3: Initialize the ranker and run it
from transformer_ranker import TransformerRanker

ranker = TransformerRanker(dataset=dataset)

result = ranker.run(models=models, batch_size=64)

# Step 4: Review Ranked Models
print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: 2.6056
Rank 2. bert-base-uncased: 2.5413
Rank 3. google/electra-small-discriminator: 1.9119
Rank 4. prajjwal1/bert-tiny: 1.4531
"""
