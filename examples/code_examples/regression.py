from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load a regression dataset. I could not find a better one yet :s
dataset = load_dataset('SetFit/stsb')

# This dataset has two columns for texts and one for float labels
text_column, text_pair_column, label_column = "text1", "text2", "label"

# You can test on cpu using smaller models
models = ['prajjwal1/bert-tiny',
          'google/electra-small-discriminator',
          'microsoft/deberta-v3-small',
          'bert-base-uncased',
          ]

# Initialize the ranker
ranker = TransformerRanker(dataset=dataset,
                           text_column=text_column,
                           text_pair_column=text_pair_column,
                           label_column=label_column,  # label column that holds floats
                           )

# ... and run it
result = ranker.run(models=models,
                    batch_size=32,
                    estimator="logme",  # use logme for regression tasks
                    )

# Print the scores
print(result)

"""Result:
Rank 1. microsoft/deberta-v3-small: -1.1082
Rank 2. google/electra-small-discriminator: -1.2708
Rank 3. bert-base-uncased: -1.3015
Rank 4. prajjwal1/bert-tiny: -1.7165
"""
