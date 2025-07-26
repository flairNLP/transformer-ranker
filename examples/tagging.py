from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load WNUT-17 (NER on tweets)
dataset = load_dataset("leondz/wnut_17")

# Define models to rank
language_models = [
    "prajjwal1/bert-tiny",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "bert-base-uncased",
]

# Initialize ranker for sequence labeling
ranker = TransformerRanker(
    dataset=dataset,
    dataset_downsample=0.2,
    label_column="ner_tags"
)

# Run ranking
result = ranker.run(language_models, batch_size=64)
print(result)
