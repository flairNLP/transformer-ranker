# Tutorial 1: Library Walkthrough

In this tutorial, we do a walkthrough of the main concepts and parameters in TransformerRanker. 
This should be the first tutorial you do.

Generally, finding the best LM for a specific task involves the following four steps: 

1. [Loading Datasets](#step-1-loading-datasets): Each task has a dataset. Load it from the Datasets library.
2. [Preparing Language Models](#step-2-preparing-language-models): TransformerRanker requires a list of language models to rank.
You provide this list. 
3. [Ranking Language Models](#step-3-ranking-language-models): Once the dataset and LM options are provided, you can now execute the ranking.
4. [Interpreting Results](#step-4-interpreting-the-results): When ranking is complete, you can select the best-suited model(s).

The goal of this tutorial is to understand these four steps. 

## Example Task 

For this tutorial, we use the example task of text classification over the classic TREC dataset. Our goal is
to find the best-suited language model. The full code for ranking LMs on TREC is:

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

In this example, we use the TREC dataset, which categorizes questions based on the type of information they seek. It comes with coarse and fine-grained categoaries:

- **Coarse-grained:** descriptions (DESC), entities (ENTY), abbreviations (ABBR), humans (HUM), locations (LOC), and numeric values (NUM). For example, the question _"What is a Devo hat?"_ is categorized under descriptions (DESC).
- **Fine-grained:** Divides broad categories into 50 subclasses, with the same question having a label DESC:def (definition).

Here's how to laod TREC:

```python
from datasets import load_dataset

# Load the 'trec' dataset
dataset = load_dataset('trec')

print(dataset)
```

Inspect the dataset structure on the [dataset page](https://huggingface.co/datasets/trec) or by printing it:

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
- __Dataset size__: Check the number of texts (around 6,000). Use this to set an appropriate `dataset_downsample` ratio for ranking.
- __Text and label columns__: Ensure the dataset has texts and labels. Some datasets can be messy.

## Step 2. Preparing Language Models

Next, prepare a list of language models to rank.
You can choose any models from the [model hub](https://huggingface.co/models).
If unsure where to start, use our predefined list of popular models:

```python
from transformer_ranker import prepare_popular_models

# Load a list of popular base-sized models
language_models = prepare_popular_models('base')

# Show the first five models
print(language_models[:5])
```

The `language_models` list contains identifiers for each model:

```console
['distilbert-base-cased', 'typeform/distilroberta-base-v2', 'bert-base-cased', 'SpanBERT/spanbert-base-cased', 'roberta-base']
```

Feel free to create your own list of models. 
We suggest exploring models that vary in pretraining tasks (masked language modeling, replaced token detection or sentence-transformers) 
and those trained with different data (multilingual, domain-specific models).

## Step 3. Ranking Language Models

You have now selected a task with its dataset (TREC) and a list of LMs to rank. 

In most cases, you can use our ranker with default parameters. Often, it is more efficient to downsample the data a bit to speed up ranking: 

```python
from transformer_ranker import TransformerRanker

# initialize ranker with dataset
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# run the ranker over the list of language models
results = ranker.run(language_models, batch_size=64)
print(results)
```

In this example, we downsampled the data to 20% and are running the ranker with a batch size of 64. You can modify these
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

The ranker prints logs to help you understand what happens as it runs.
It iterates over each model and (1) embeds texts, (2) scores embeddings using an estimator.
Logs show which model is currently being assessed.

```bash
transformer_ranker:Text and label columns: 'text', 'coarse_label'
transformer_ranker:Task type: 'text classification'
transformer_ranker:Dataset size: 1190 (downsampled to 0.2)
Computing Embeddings:  100%|██████████| 19/19 [00:02<00:00,  7.69it/s]
Transferability Score: 100%|██████████| 1/1 [00:00<00:00,  1.01it/s]
transformer_ranker:distilbert-base-cased, estimated score: 3.9598
Computing Embeddings:  100%|██████████| 19/19 [00:00<00:00, 21.38it/s]
Transferability Score:  70%|███████   | 1/1 [00:00<00:00,  9.15it/s]
```

Running time varies based on dataset size and selected language models. Here are two examples:

- The **downsampled TREC** dataset (1,190 instances) takes about 2.3 minutes to process 17 base-sized models: 1.2 minutes for downloading and 1.1 minutes for embedding and scoring.
- The full TREC dataset (5,952 instances) takes about 4.8 minutes: 1.2 minutes for downloads and 3.6 minutes for embedding and scoring.

We used Colab Notebook with a Tesla T4 GPU. Note that TREC has short texts (10 words on average) and embedding longer texts will take more time.

## Step 4. Interpreting the Results

Doing `print(results)` displays the ranked language models from Step 2, along with their **transferability scores**.  
A **higher score** means the model is better suited for your dataset.
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

The top-ranked model _'deberta-v3-base'_ is a strong candidate for fine-tuning. We recommend fine-tuning other highly ranked models for comparison.

To fine-tune the top-ranked model, use any framework of your choice (e.g. 
<a href="https://flairnlp.github.io/">Flair</a> or Transformers — we opt for the first one ;p).

## Summary

This tutorial shows the four steps for selecting the best-suited LM for an NLP task.
We (1) loaded a text classification dataset, (2) prepared a list of language model names, and (3) ranked them based on transferability scores. 

In the next tutorial, we give examples for a variety of NLP tasks.

