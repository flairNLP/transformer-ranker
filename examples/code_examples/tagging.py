from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load the 'conll2003' dataset
dataset = load_dataset('conll2003')

# Use smaller models to test on cpu
models = ['prajjwal1/bert-tiny',
          'google/electra-small-discriminator',
          'microsoft/deberta-v3-small',
          'bert-base-uncased',
          ]

# Initialize the ranker
ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2)

# ... and run it
result = ranker.run(models=models, batch_size=64)

# Print the scores
print(result)

"""Result
Rank 1. microsoft/deberta-v3-small: 2.6497
Rank 2. bert-base-uncased: 2.5935
Rank 3. google/electra-small-discriminator: 1.9449
Rank 4. prajjwal1/bert-tiny: 1.4887
"""

