# Text Classification

This markdown explains how to use the library to choose the best-suited language model for text classification datasets.
We will load a sample dataset and rank 17 models using transferability metrics.
Let's follow these steps:

1. [Loading Datasets](#1-loading-and-inspecting-datasets): Load a text classification dataset using the Datasets library.
2. [Preparing Language Models](#2-preparing-language-models): Choose from our 17 language models, or create your own custom list.
3. [Ranking Language Models](#3-ranking-language-models): Rank the selected models on a downsampled part (20%) of the dataset.
4. [Interpreting Results](#4-result-interpretation): Review the scores to find the best-suited model, offering a good starting point for fine-tuning.

<details>
<summary>
Complete code for ranking language models on TREC<br>
</summary>

```python3
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

</details>

## Setup and Installation

First, make sure Python 3.8 or later is installed. Install the ranker package using pip:

```
pip install transformer-ranker
```

## 1. Loading and Inspecting Datasets

Use Hugging Face’s Datasets library to load and access various text datasets.
You can explore datasets in the [text classification](https://huggingface.co/datasets?task_categories=task_categories:text-classification&sort=trending) section on Hugging Face.

In this example, we use the TREC question classification dataset, which groups questions by type of information asked.
It comes with coarse and fine-grained question categories:

- **Coarse-grained classes:** Six broad categories, including descriptions, entities, abbreviations, humans, locations, and numeric values. For example, the question _"What is a Devo hat?"_ falls under the coarse class DESC (description and abstract concept).
- **Fine-grained classes:** Splits broad categories into 50 subclasses, where the question _"What is a Devo hat?"_ belongs to a finer class, DESC:def (definition).

Here's how to load TREC:

```python
from datasets import load_dataset

# Load the 'trec' dataset
dataset = load_dataset('trec')

print(dataset)
```

It's helpful to inspect the dataset structure on the [web interface](https://huggingface.co/datasets/trec) or by printing it out:

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

Key details to note:
- __Dataset size__: Check the number of texts (around 6,000). This will help set a good `dataset_downsample` ratio for ranking.
- __Text and label columns__: Ensure the dataset includes texts and labels. Some datasets might be incomplete due to the absence of quality control during uploads. TREC has _'text'_, _'coarse_label'_, and _'fine_label'_ columns, making it ready for text classification.

## 2. Preparing Language Models

Next, prepare a list of language models to assess for the downstream task.
You can choose any models from the [model hub](https://huggingface.co/models).
If unsure where to start, use the predefined list of popular models:

```python3
from transformer_ranker import prepare_popular_models

# Load a list of popular base-sized models
language_models = prepare_popular_models('base')

# Show the first five models
print(language_models[:5])
```

The `language_models` list contains identifiers for each model:

```bash
['distilbert-base-cased', 'typeform/distilroberta-base-v2', 'bert-base-cased', 'SpanBERT/spanbert-base-cased', 'roberta-base']
```

We recommend trying models with different pretraining tasks (e.g., masked language modeling, replaced token detection, sentence transformers)
or models trained on diverse data (e.g., multilingual or domain-specific models).

## 3. Ranking Language Models

With the dataset and models ready, the next step is ranking.
Initialize the ranker with your dataset and set any dataset-specific parameters (e.g., downsampling ratio):

```python3
from transformer_ranker import TransformerRanker

ranker = TransformerRanker(dataset, dataset_downsample=0.2)
```

Key parameters to consider:
- `dataset_downsample` (0.2): Reduces dataset size for faster ranking. The ranker will log the reduced number of texts as: _"Dataset size: 1190 for TREC (downsampled to 0.2)"_.
- `text_column` (optional): The name of the column containing texts (e.g. sentences, documents, words).
- `label_column` (optional): The name of the column for labels. Labels can be strings, integers, or floats for regression tasks. For TREC’s fine-grained categories, set this to `label_column=fine_label`.
- `text_pair_column` (optional): For tasks that involve text pairs, specify the second text column.

Run the ranker with your list of language models:

```python3
results = ranker.run(language_models, batch_size=64)
print(results)
```

- `batch_size` (64): Sets how many texts are processed per batch during embedding. Since models aren't fine-tuned, larger batch sizes (e.g., 64 or 128) can be used. If memory issues occur, reduce the batch size.

<details>
<summary>
<em>Note</em>: Different-sized models may need different batch sizes.<br>
</summary>

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker

dataset = load_dataset('trec')

ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2)

# Step 1: Rank small models
small_models = ['prajjwal1/bert-tiny', 'google/electra-small-discriminator']

# ... using a large batch size
result = ranker.run(models=small_models, batch_size=124)

# Step 2: Add rankings of larger models
large_models = ['bert-large-cased', 'google/electra-large-discriminator']

## ... using a small batch size
result.append(ranker.run(batch_size=16, models=large_models))

# Look at merged results
print(result)
```

</details>

The ranker logs steps to help you understand what happens as it runs.
It iterates over each model and (1) embeds texts, (2) scores embeddings using an estimator.
Logs show which model is currently being assessed.

```bash
transformer_ranker:Text and label columns: 'text', 'coarse_label'
transformer_ranker:Task type: 'sentence classification'
transformer_ranker:Dataset size: 1190 (downsampled to 0.2)
Retrieving Embeddings: 100%|██████████| 19/19 [00:02<00:00,  7.69it/s]
Scoring Embeddings:    100%|██████████| 1/1 [00:00<00:00,  1.01it/s]
transformer_ranker:distilbert-base-cased, estimated score: 3.9598
Retrieving Embeddings: 100%|██████████| 19/19 [00:00<00:00, 21.38it/s]
Scoring Embeddings:    100%|██████████| 1/1 [00:00<00:00,  8.93it/s]
transformer_ranker:typeform/distilroberta-base-v2, estimated score: 3.8139
Retrieving Embeddings: 100%|██████████| 19/19 [00:01<00:00, 11.92it/s]
Scoring Embeddings:     70%|███████   | 1/1 [00:00<00:00,  9.15it/s]
```

Running time varies based on dataset size and language models. Here are two examples:

- For the **downsampled TREC** dataset (1,190 instances), scoring 17 base-sized models takes approximately 2.3 minutes—1.2 minutes to download the models and 1.1 minutes for embedding and scoring.
- For the **full TREC** dataset (5,952 instances), scoring the same 17 models takes around 4.8 minutes—1.2 minutes for downloading and 3.6 minutes for embedding and scoring.

We used a GPU-enabled Colab Notebook with a Tesla T4.
Keep in mind that TREC has short questions, averaging about 10 words each.
For longer documents, embedding and scoring takes more time.

## 4. Result Interpretation

The results are sorted in descending order.
Transferability scores show how well each model suits your task.
Higher scores indicate better suitability for the dataset.
Here’s the output after ranking 17 language models on TREC:

```bash
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

The model '_deberta-v3-base_' ranks the highest, making it a good starting point for fine-tuning.
However, we recommend fine-tuning other highly ranked models for comparison.

## Summary

This example showed how to use `transformer-ranker` to select the best-suited language model for the TREC question classification dataset.
We (1) loaded a text classification dataset, (2) selected language models, and (3) ranked them based on transferability scores.
Now you can decide which models to fine-tune and which to skip.

To fine-tune the top-ranked model, use a framework of your choice (e.g. Flair or Transformers—we opt for the first one ;p).
