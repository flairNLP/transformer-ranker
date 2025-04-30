# Tutorial 3: Advanced

In the advanced tutorial, we go over how to change transferability metrics using the estimator parameter, load custom datasets, and run TransformerRanker with non-default settings. We also show a special case: finding the best-performing layer in a single language model.

## Loading Custom Datasets

Not all datasets are available in the Hugging Face Datasets library. If you have a custom dataset stored in local text files, you can load it using the following snippet:

```python
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load dataset from text files
dataset = load_dataset(
    "text",
    data_files={
        "train": "path/to/dataset/train.txt",
        "dev": "path/to/dataset/dev.txt",  # optional
        "test": "path/to/dataset/test.txt",  # optional
    },
)

# Prepare language models
language_models = prepare_popular_models('base')

# Initialize the ranker with the dataset
ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2)

# ... and run it
results = ranker.run(models=language_models, batch_size=32)
```

Specifying train/dev/test is optional—TransformerRanker merges and downsamples datasets automatically.
Once loaded, do the LM ranking as in previous tutorials. 

To load .json or .csv files take a look at the [guide](https://huggingface.co/docs/datasets/v1.7.0/loading_datasets.html#from-local-files) of Datasets.

## Transferability Metrics

The transferability metric can be changed by setting the `estimator` parameter in the .run() method. To change the metric to LogME, do:

```python
results = ranker.run(language_models, estimator="logme")
```

__Transferability Explanation:__ Transferability metrics estimate how well a model is likely to perform on a new dataset without requiring fine-tuning. For a pre-trained language model, this means evaluating how well its embeddings capture the structure of the target dataset.

The following metrics are supported:

- **`hscore`** *(default)*: Fast and generally the best choice for most datasets. Suited for classification tasks
  [View source](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/hscore.py).

- **`logme`**: Suitable for both classification and regression tasks
  [View source](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/logme.py).

- **`nearestneighbors`**: Slowest and least accurate, but easy to interpret
  [View source](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/nearesneighbors.py).

For a better understanding of each metric, see our code and comments, or refer to the original papers. 

## Layer Aggregation

To improve existing transferability estimation approaches, we propose to average all hidden layers in LMs. This can be controled by changing the `layer_aggregator` parameter.

To use the best performing layer, do:

```python
results = ranker.run(language_models, layer_aggregator="bestlayer")
```

Here are the supported methods:

- `layermean` (default): Averages of all hidden states in a language model.
- `bestlayer`: Scores each hidden state and uses the layer with the highest transferability score.
- `lastlayer`: Uses the last hidden state as embeddings of a language model. 

## Example: LM Ranking Using Custom Settings on CoNLL2003

Here’s how to rank language models using custom settings: `logme` as the estimator and `bestlayer` as the layer aggregator:

```python
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load the CoNLL-03 dataset from HuggingFace
dataset = load_dataset('conll2003')

# Prepare a list of popular 'base' LMs as candidates
language_models = prepare_popular_models('base')

# Initialize the ranker with the dataset
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# ... and run it with custom settings
results = ranker.run(language_models, estimator="logme", layer_aggregator="bestlayer")
```

The output should look as follows:

```python
print(results)
```

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

Compare this ranking with the one in the main [README](https://github.com/flairNLP/transformer-ranker?tab=readme-ov-file#example-2-really-find-the-best-lm).


## Example: Inspecting Layer Transferability in a Single LM

You can also inspect layer-wise transferability scores for a single large model.
Here’s how to rank layers of DeBERTa-v2-xxlarge (1.5B) on CoNLL2003:


```python
from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load the CoNLL dataset
conll = load_dataset('conll2003')

# Use a single language model
language_model = ['microsoft/deberta-v2-xxlarge']

# Initialize the ranker with the dataset
ranker = TransformerRanker(dataset=conll, dataset_downsample=0.2)

# ... and run it with custom settings
result = ranker.run(models=language_model, layer_aggregator='bestlayer')
```

This returns the best-scoring layer along with a dictionary of all layers:

```console
transformer_ranker.ranker:microsoft/deberta-v2-xxlarge, hscore: 2.8912 (layer -41)
layer scores: {-1: 2.7377, -2: 2.8024, -3: 2.8312, -4: 2.8270, ..., -48: 2.5153}
```

Useful for inspecting layer-wise transferability for a downstream dataset.

## Summary

In this tutorial, we explored advanced features of TransformerRanker: how to load custom datasets, switch transferability metrics with the `estimator` parameter, and identify the best-suited layer using the `layer_aggregator` parameter. These settings can be adjusted based on the task or to compare different transferability metrics. 
