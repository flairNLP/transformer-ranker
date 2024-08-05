# TransformerRanker - Text Classification

# Step 1: Load a text classification dataset
from datasets import load_dataset

dataset = load_dataset('trec')

# You can specify exact column names, but this is generally unnecessary for most hf classification datasets.
text_column, label_column = "text", "coarse_label"

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
                           label_column=label_column,
                           )

result = ranker.run(models=models, batch_size=64)

# Step 4: Review Ranked Models
print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: 3.5737
Rank 2. bert-base-uncased: 3.4666
Rank 3. google/electra-small-discriminator: 2.6852
Rank 4. prajjwal1/bert-tiny: 1.9463
"""
