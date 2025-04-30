# Tutorial 1: Library Walkthrough

In this tutorial, we do a walkthrough of the main concepts and parameters in TransformerRanker.

Generally, finding the best LM for a specific tasks involves the following steps:

1. [Loading Datasets](#step-1-loading-datasets): Each task has a dataset. Load it from the Datasets library.
2. [Preparing Language Models](#step-2-preparing-language-models): TransformerRanker requires a list of language models to rank.
You provide this list. 
1. [Ranking Language Models](#step-3-ranking-language-models): Once the dataset and LM options are provided, you can now execute the ranking.
2. [Interpreting Results](#step-4-interpreting-the-results): When ranking is complete, you can select the best-suited model(s) for the dataset.

The goal of this tutorial is to understand these four steps. 

## Example Task 

We use the example task of text classification over the classic TREC dataset. Our goal is to find the best-suited LM for TREC. The full code:

```python
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load a dataset
dataset = load_dataset('trec')

# Load a predefined models or create your own list of models
language_models = prepare_popular_models('base')

# Initialize the ranker with the dataset
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# Run it with your models
results = ranker.run(language_models, batch_size=64)

# Inspect results
print(results)
```

## Step 1. Loading Datasets

Use the Hugging Face Datasets library to load datasets from their [text classification](https://huggingface.co/datasets?task_categories=task_categories:text-classification&sort=trending) section. You load a dataset by passing its string identifier.

Here is how to load TREC:

```python
from datasets import load_dataset

# Load the 'trec' dataset
dataset = load_dataset('trec')

print(dataset)
```

Inspect the dataset by printing it:

```bash
DatasetDict({
    train: Dataset({
        features: ['text', 'coarse_label', 'fine_label'],
        num_rows: 5452
    })
    test: Dataset({
        features: ['text', 'coarse_label', 'fine_label'],
        num_rows: 500
    })
})
```

Key things to note:
- __Dataset size__: TREC has ~6,000 texts. Use this to set a `downsample` ratio.
- __Text and label fields__: Some datasets are messy. Ensure texts and labels are non-empty. Note that some datasets may have multiple label fields (e.g., coarse and fine-grained classes).

## Step 2. Preparing Language Models

Next, prepare a list of language models to rank.
You can choose any models from the [model hub](https://huggingface.co/models).
If unsure where to start, use our predefined list of models:

```python
from transformer_ranker import prepare_popular_models

# Load a list of popular base-sized models
language_models = prepare_popular_models('base')

# Show the first five models
print(language_models[:5])
```

The `language_models` list contains string identifiers for each model:

```console
['distilbert-base-cased', 'typeform/distilroberta-base-v2', 'bert-base-cased', 'SpanBERT/spanbert-base-cased', 'roberta-base']
```

Feel free to create your own list of model names.
We recommend including models that were pre-trained on different tasks and datasets.

## Step 3. Ranking Language Models

You have now selected a task with its dataset (TREC) and a list of LMs to rank. 

In most cases, you can use our ranker with default parameters. Often, it is more efficient to downsample the dataset to speed up ranking: 

```python
from transformer_ranker import TransformerRanker

# initialize ranker with dataset
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# run the ranker over the list of language models
results = ranker.run(language_models, batch_size=64)
print(results)
```

Here we downsampled the data to 20% and are running the ranker with a batch size of 64. You can modify these
two parameters: 
- `dataset_downsample`: Set it to 1. to estimate over the full dataset. Or lower than 0.2 to make an estimation even faster. 
We found that downsampling to 20% often does not hurt estimation performance.
- `batch_size`: Set it higher or lower depending on your GPU memory. Only big GPUs can handle a large batch size.

<details>
<summary>
<em>Advanced</em>: Different-sized models may need different batch sizes.<br>
</summary>

```python
from datasets import load_dataset
from transformer_ranker import TransformerRanker

dataset = load_dataset('trec')

ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2)

# Step 1: Rank small models
small_models = ['prajjwal1/bert-tiny', 'google/electra-small-discriminator']

# ... using a large batch size
result = ranker.run(models=small_models, batch_size=128)

# Step 2: Add rankings of larger models
large_models = ['bert-large-cased', 'google/electra-large-discriminator']

## ... using a small batch size
result.append(ranker.run(batch_size=16, models=large_models))

# Look at merged results
print(result)
```

</details>

### Optional: Specifying Labels

***Note:*** TREC has two sets of labels (fine-grained and coarse-grained). By default, TransformerRanker heuristically 
determines which field in the dataset is the label to use. In the case of TREC, it 
automatically uses the coarse-grained labels. 

But you can also directly indicate which field to use as labels by passing the `label_column`.
For instance, if instead you want to find 
the best LM for **fine-grained** question classification, use the following code: 

```python
from transformer_ranker import TransformerRanker

# initialize ranker with dataset and indicate the label column
ranker = TransformerRanker(dataset, label_column='fine_label', dataset_downsample=0.2)

# run the ranker over the list of language models
results = ranker.run(language_models, batch_size=64)
print(results)
```

### Running the Ranker

When running the ranker, each LM is processed individually:
TransformerRanker embeds the texts with the LM and scores them using a transferability metric.
The log shows which LM is currently being assessed:

```bash
transformer_ranker:Text and label columns: 'text', 'coarse_label'
transformer_ranker:Task type: 'text classification'
transformer_ranker:Dataset size: 1190 (downsampled to 0.2)
Computing Embeddings:  100%|██████████| 19/19 [00:02<00:00,  7.69it/s]
Transferability Score: 100%|██████████| 1/1 [00:00<00:00,  1.01it/s]
transformer_ranker:distilbert-base-cased, hscore: 3.9598
Computing Embeddings:  100%|██████████| 19/19 [00:00<00:00, 21.38it/s]
Transferability Score:  70%|███████   | 1/1 [00:00<00:00,  9.15it/s]
```

Ranking is generally fast, but runtime depends on dataset size, text length, and the size of selected models.
For example, on TREC:

- ~2.3 min to rank 17 base models on 20% of the dataset (1190 texts)
- ~4.8 min to rank same models on the full dataset (5952 texts)

Tested on a Colab Notebook (Tesla T4 GPU).

## Step 4. Interpreting the Results

Once the ranking is complete, the final list of LM names and their **transferability scores** wil be shown. 
Higher transferability means better suitability for the dataset.
The final output of the TREC example is:

```console
Rank 1. microsoft/deberta-v3-base: 4.0172
Rank 2. google/electra-base-discriminator: 4.0068
Rank 3. microsoft/mdeberta-v3-base: 4.0028
Rank 4. distilbert-base-cased: 3.9598
Rank 5. Twitter/twhin-bert-base: 3.9394
Rank 6. bert-base-cased: 3.9241
Rank 7. sentence-transformers/all-mpnet-base-v2: 3.9138
Rank 8. roberta-base: 3.9028
Rank 9. Lianglab/PharmBERT-cased: 3.8861
Rank 10. FacebookAI/xlm-roberta-base: 3.8665
Rank 11. SpanBERT/spanbert-base-cased: 3.829
Rank 12. typeform/distilroberta-base-v2: 3.8139
Rank 13. dmis-lab/biobert-base-cased-v1.2: 3.8125
Rank 14. german-nlp-group/electra-base-german-uncased: 3.8005
Rank 15. KISTI-AI/scideberta: 3.7468
Rank 16. sentence-transformers/all-MiniLM-L12-v2: 3.4271
Rank 17. google/electra-small-discriminator: 2.9615
```

Where the top-ranked model is _'deberta-v3-base'_.
This should be the LM to use for the selected downstream dataset.
However, we recommend fine-tuning other highly ranked models for comparison.

To fine-tune the top-ranked model, use any framework of your choice (e.g. 
<a href="https://flairnlp.github.io/">Flair</a> or Transformers — we opt for the first one ;p).

## Summary

This tutorial showed how to use TransformerRanker in four steps. We loaded a text classification dataset, prepared a list of LM names, and ranked them based on transferability scores. 

In the next tutorial, we give examples for a variety of NLP tasks.
