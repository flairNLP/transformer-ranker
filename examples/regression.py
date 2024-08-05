# Transformer Ranker - Text Regression

# Step 1: Load a dataset using transformer datasets
from datasets import load_dataset

# Load the 'SetFit/stsb' dataset used for the Semantic Textual Similarity Benchmark (STS-B) task
dataset = load_dataset('SetFit/stsb')

# You can specify exact column names, but this is generally unnecessary for most hf classification datasets.
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
                           text_pair_column=text_pair_column,
                           label_column=label_column,
                           )

result = ranker.run(models=models,
                    estimator="logme",  # use logme for regression tasks
                    batch_size=32,
                    )

# Step 4: Review Ranked Models
print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: -1.8326
Rank 2. google/electra-small-discriminator: -1.8723
Rank 3. bert-base-uncased: -1.8767
Rank 4. prajjwal1/bert-tiny: -2.0059
"""
