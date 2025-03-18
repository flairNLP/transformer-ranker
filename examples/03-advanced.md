# Tutorial 3: Advanced

Previous tutorials demonstrated ranking models with standard datasets and default parameters in the TransformerRanker.
This tutorial moves further by showing how to load custom datasets not available on the Hugging Face Hub.
Then, we introduce two optional parameters: the transferability metric and the layer aggregation method.

## Loading Custom Datasets

TransformerRanker uses the `load_dataset()` method from the **Datasets** library.
To load a dataset from local files instead of the hub, prepare your dataset as `.txt` files in a directory
and load it as follows:

```python
from datasets import load_dataset

dataset = load_dataset(
    "text",
    data_files={
        "train": "path/to/dataset/train.txt",
        "dev": "path/to/dataset/dev.txt",  # optional
        "test": "path/to/dataset/test.txt",  # optional
    },
)
```

Splitting into train/dev/test is optional—TransformerRanker automatically merges and downsamples datasets for ranking.

For `.csv` or `.json` datasets, refer to the official
[load_datasets() guide](https://huggingface.co/docs/datasets/v1.7.0/loading_datasets.html#from-local-files). 

Once loaded, initialize TransformerRanker with your dataset as shown in previous tutorials.

## Setting the Transferability Metric

Transferability metric can be changed using the `estimator` parameter in the `.run()` method.
The supported metrics are: 

- **H-score (default):** fast and accurate for most datasets, suited for classification tasks.
- **LogME:** Suited for both classification and regression tasks.
- **Nearest Neighbors:** Slowest and least accurate, but easy to interpret.

To change the metric to LogME, do this then running the ranker:

```python
result = ranker.run(language_models, estimator="logme")
```

For better understanding of each metric, see [code and comments](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/).


## Setting the Layer Aggregation

Another parameter that can influence the ranking quality is the `layer_aggegator='layermean'`. 
This can optionally be changed to one of the following:

- **Last layer** uses the last hidden state as embeddings from each language model.
- **Layer mean** uses the average of all hidden states as an embedding in a language model.
- **Best layer** uses the layer that results in a highest transferability score. 

Here's how to set the aggregation to `bestlayer`:

```python
result = ranker.run(language_models, estimator="logme", layer_aggregator="bestlayer")
```

The default settings are `hscore` for estimator and `layermean` for layer aggregation (which generally perform well).
Some datasets can benefit from different configurations.

## Example: Ranking on CoNLL2003 using custom parameters

Below, we show the use of `logme` and `bestlayer` with the CoNLL2003 dataset:

```python
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load the CoNLL-03 dataset from HuggingFace
dataset = load_dataset('conll2003')

# Prepare a list of popular 'base' LMs as candidates
language_models = prepare_popular_models('base')

# Initialize the ranker with the dataset
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# Run the ranker with custom settings
results = ranker.run(language_models, estimator="logme", layer_aggregator="bestlayer")

# Display the ranking results
print(results)
```

The output should look as follows:

```console
Rank 1. microsoft/mdeberta-v3-base: 0.7566
Rank 2. microsoft/deberta-v3-base: 0.7565
Rank 3. typeform/distilroberta-base-v2: 0.7165
Rank 4. google/electra-base-discriminator: 0.7115
Rank 5. roberta-base: 0.702
Rank 6. sentence-transformers/all-mpnet-base-v2: 0.6895
Rank 7. FacebookAI/xlm-roberta-base: 0.689
Rank 8. bert-base-cased: 0.6843
Rank 9. Twitter/twhin-bert-base: 0.6685
Rank 10. german-nlp-group/electra-base-german-uncased: 0.6088
Rank 11. sentence-transformers/all-MiniLM-L12-v2: 0.5829
Rank 12. distilbert-base-cased: 0.5685
Rank 13. Lianglab/PharmBERT-cased: 0.5365
Rank 14. google/electra-small-discriminator: 0.5128
Rank 15. KISTI-AI/scideberta: 0.4777
Rank 16. SpanBERT/spanbert-base-cased: 0.4299
Rank 17. dmis-lab/biobert-base-cased-v1.2: 0.3931
```

Compare this ranking with the one in the main __README__.

## Summary

Here, we demonstrated how to load a custom dataset not hosted on the Hugging Face Hub.
We then introduced two optional parameters for TransformerRanker: `estimator` and `layer_aggregator`,
which can be adjusted based on the task or used to compare different transferability metrics.
