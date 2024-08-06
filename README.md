# TransformerRanker

A lightweight library to efficiently rank transformer language models for classification tasks.

There is a multitude of pre-trained language models available.
Fine-tuning each to select which one scores best on your classification dataset is both time and resource expensive. 
TransformerRanker is a library that can be used for the model selection process, where you can choose any dataset from the HuggingFace collection of [datasets](https://huggingface.co/datasets),
select different model candidates from the [model hub](https://huggingface.co/models), and let the tool rank them using _transferability estimation_ metrics.

## Installation

You can install the tool using pip:

```python3
pip install transformer-ranker
```

## Three-step-interface

### Step 1. Load your dataset

Choose any dataset from the [datasets](https://huggingface.co/docs/datasets/en/index) library:
```python3
from datasets import load_dataset

# Load your dataset using hf loader
dataset = load_dataset('conll2003')
```

Take a look how to [load your custom](https://huggingface.co/docs/datasets/v1.1.1/loading_datasets.html#from-local-files) dataset using HuggingFace datasets.

### Step 2. Prepare a list of language models

Choose any model names from the [model hub](https://huggingface.co/models):

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

### Step 3. Rank Models

Initialize the ranker with your dataset and run it your models:

```python3
from transformer_ranker import TransformerRanker

# Initialize the ranker with your dataset
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# Run it with selected transformer models
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

Using these results you can exclude the lower-ranked models to only focus on the top-ranked models for further exploration.

## License

[MIT](LICENSE)