# Tutorial 3: Advanced

Previous tutorials showed how to rank LMs using default parameters and datasets from the hub.
This one covers how to load custom datasets and use two optional parameters in the ranker: `estimator` and `layer_aggregator`.

## Loading Custom Datasets

TransformerRanker uses `load_dataset()` from the 🤗 Datasets library.
To load local text files instead of datasets from the hub, do:

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

Train/dev/test splits are optional—TransformerRanker merges and downsamples datasets automatically. Once loaded,
initialize the ranker with your dataset as shown in previous tutorials. For `.csv` or `.json` formats, see the complete
load_dataset() [guide](https://huggingface.co/docs/datasets/v1.7.0/loading_datasets.html#from-local-files).

## Transferability Metric

Transferability metric can be changed by setting the `estimator` parameter in the `.run()` method.
For example, to switch to the LogME, do:

```python
results = ranker.run(language_models, estimator="logme")
```

__Transferability Explanation:__ Transferability metrics estimate how well a model can apply its knowledge to a new task without fine-tuning.
For a pre-trained LM, this means assessing how well its embeddings align with a new dataset.

Here are the supported metrics:

- `hscore` (default): Fast and generally the best choice for most datasets, suited for classification tasks [H-Score code](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/hscore.py).
- `logme`: Suitable for both classification and regression tasks [LogME code](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/logme.py).
- `nearestneigbors`: Slowest and least accurate, but easy to interpret [k-NN code](https://github.com/flairNLP/transformer-ranker/blob/main/transformer_ranker/estimators/nearesneighbors.py).

For a better understanding of each metric, refer to the comments in the code or see the original papers. 

## Layer Aggregation

Layer aggregation method can be changed by setting the `layer_aggregator` parameter in the `.run()` method.
For example, to use the best performing layer, do:

```python
results = ranker.run(language_models, layer_aggregator="bestlayer")
```

Here are the supported methods:

- `layermean` (default): Averages of all hidden states in a language model.
- `bestlayer`: Scores each hidden state and uses the layer with the highest transferability score.
- `lastlayer`: Uses the last hidden state as embeddings of a language model. 

## Example: CoNLL2003 with Custom Settings

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

## Summary

Here, we demonstrated how to load a custom dataset not hosted on the Hugging Face Hub.
We then introduced two optional parameters for TransformerRanker: `estimator` and `layer_aggregator`,
which can be adjusted based on the task or used to compare different transferability metrics.
