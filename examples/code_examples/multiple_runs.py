from datasets import load_dataset
from transformer_ranker import TransformerRanker

dataset = load_dataset('trec')

ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2)

# Load smaller models
models = ['prajjwal1/bert-tiny', 'google/electra-small-discriminator']

# ... and rank them using a large batch size
result = ranker.run(models=models, batch_size=124)
print(result)

# Add larger models
models = ['bert-large-cased', 'google/electra-large-discriminator']

# ... and rank them using a small batch size
result.append(ranker.run(batch_size=16, models=models))

print(result)

"""Result:
Rank 1. google/electra-large-discriminator: 4.2713
Rank 2. bert-large-cased: 4.1123
Rank 3. google/electra-small-discriminator: 2.9056
Rank 4. prajjwal1/bert-tiny: 1.956
"""
