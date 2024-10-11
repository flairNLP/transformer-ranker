from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load 'trec' dataset
dataset = load_dataset('trec')

# Add column names, but that's not necessary
text_column, label_column = "text", "coarse_label"

# You can test on cpu using smaller models
language_models = ['prajjwal1/bert-tiny',
                   'google/electra-small-discriminator',
                   'microsoft/deberta-v3-small',
                   'bert-base-uncased',
                   ]

# Initialize the ranker
ranker = TransformerRanker(dataset=dataset,
                           text_column=text_column,
                           label_column=label_column,
                           dataset_downsample=0.2,
                           )

# ... and run it
result = ranker.run(models=language_models, batch_size=32)

# Print the scores
print(result)

"""Result: 
Rank 1. microsoft/deberta-v3-small: 4.0091
Rank 2. bert-base-uncased: 3.8054
Rank 3. google/electra-small-discriminator: 2.959
Rank 4. prajjwal1/bert-tiny: 1.9793
"""
