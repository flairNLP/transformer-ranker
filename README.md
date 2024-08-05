# TransformerRanker

A lightweight library to efficiently rank transformer language models for classification tasks

Choosing the right model for your classification dataset can be costly.
We keep it simple by leveraging _transferability estimation_, eliminating the need for extensive fine-tuning.
This tool comes with an intuitive three-step interface, compatible with any transformer model and a classification dataset from the HuggingFace and PyTorch ecosystems.

## Example using transformers

You can install the tool using pip:

```python3
pip install transformer-ranker
```

### Step 1. Load your dataset

Choose any dataset from the [datasets](https://huggingface.co/docs/datasets/en/index) library:
```python3
from datasets import load_dataset

# Load your dataset using hf loader
dataset = load_dataset('conll2003')
```

Take a look how to load your custom dataset using datasets [here](https://huggingface.co/docs/datasets/v1.1.1/loading_datasets.html#from-local-files).

### Step 2. Prepare a list of language models

Choose any model names from the model [hub](https://huggingface.co/models):

```python3
# Prepare a list of model handles
language_models = [
    "sentence-transformers/all-mpnet-base-v2",
    "xlm-roberta-large",
    "google/electra-large-discriminator",
    "microsoft/deberta-v3-large",
    "nghuyong/ernie-2.0-large-en",
    # ...
]
```

...or use our recommended list of models to try out:

```python3 
language_models = prepare_popular_models('base')
```

### Step 3. Initialize the ranker and run it

Initialize the ranker with your dataset and run it your models:

```python3
from transformer_ranker import TransformerRanker

# Initialize the ranker with your dataset
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# Run it with your models
results = ranker.run(language_models, batch_size=64)
```

### Review ranked models:

```python3
print(results)
```

Display results showing models sorted by their transferability scores:

```bash
Rank 1. microsoft/deberta-v3-large: 2.7962
Rank 2. nghuyong/ernie-2.0-large-en: 2.7788
Rank 3. google/electra-large-discriminator: 2.7486
Rank 4. xlm-roberta-large: 2.6695
Rank 5. sentence-transformers/all-mpnet-base-v2: 2.5709
...
```

After running it, you can identify most promising models for your datasets.

## License

[MIT](LICENSE)