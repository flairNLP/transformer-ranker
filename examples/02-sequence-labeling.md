# Sequence Labeling - PoS and NER

This example shows how to rank language models for two sequence labeling tasks: Part-of-Speech tagging (PoS) and Named Entity Recognition (NER).
We will load two token classification datasets and rank 17 language models using _transferability metrics_ to find the best-suited ones.
Let's break it down into steps:

1. [Loading Datasets](#1-loading-and-inspecting-datasets): Load two token classification datasets using the Datasets library.
2. [Preparing Language Models](#2-preparing-language-models): Choose from 17 language models, or create your own custom list.
3. [Ranking Language Models](#3-ranking-language-models): Rank models for English PoS (Universal Dependencies) and NER (WNUT_17).
4. [Interpreting Results](#4-result-interpretation): Review transferability scores to select the best-suited model for each dataset.

<details> <summary>Complete code for ranking language models on UD's 'en_lines'<br> </summary>

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load UD's 'en_lines' dataset
dataset_pos = load_dataset('universal-dependencies/universal_dependencies', 'en_lines')

# Load a list of popular 'base' models
language_models = prepare_popular_models('base')

# Initialize the Ranker
ranker = TransformerRanker(dataset_pos,
                           dataset_downsample=0.5,
                           text_column="tokens",
                           label_column="upos")

# ... and run it
results = ranker.run(language_models, batch_size=64)

# Inspect results
print(results)
```

</details>

## Setup and Installation

Ensure Python 3.8 or later is installed, then install the package via pip:

```bash
pip install transformer-ranker
```

## 1. Loading and Inspecting Datasets

In this example, we use two token classification datasets. These datasets consist of sentences as lists of words, where each word has a label.

- **Part-of-Speech Tagging** A dataset from Universal Dependencies (UD) containing English subtitles annotated with parts of speech.
- **Named Entity Recognition** The WNUT_17 dataset, which features Twitter data annotated for entity recognition.

#### Part-of-Speech Tagging

```python
from datasets import load_dataset

# Load Universal Dependencies PoS dataset
dataset_pos = load_dataset('universal-dependencies/universal_dependencies', 'en_lines')

# Inspect the dataset
print(dataset_pos)
```

#### Named Entity Recognition

```python
from datasets import load_dataset

# Load NER dataset
dataset_ner = load_dataset('leondz/wnut_17')

# Inspect the dataset
print(dataset_ner)
```

When inspecting datasets, note the following:
- __Dataset size__: Both datasets have around 5,000 texts.
- __Text and label columns__: UD includes _'tokens'_ (for texts) and _'upos'_ (for PoS labels), while the WNUT_17 includes _'tokens'_ and _'ner_tags'_.

## 2. Preparing Language Models

Next, list the language models to assess.
For consistency, let's use the same list of models as in the text classification example.

```python3
from transformer_ranker import prepare_popular_models

# Load a predefined list of popular base models
language_models = prepare_popular_models('base')
```

Or try using a custom list. We recommend adding models trained on different data or with different pretraining tasks.

```python3

language_models = [
    'xlm-roberta-base',
    'facebook/bart-base',
    'johngiorgi/declutr-sci-base'
]
```

## 3. Ranking Language Models

Initialize the ranker with the dataset and specify text and label columns.
Next, run the ranking using your list of language models.

#### For PoS Tagging

```python3
from transformer_ranker import TransformerRanker

ranker = TransformerRanker(dataset_pos,
                           downsample=0.5,
                           text_column='tokens',
                           label_column='upos',  # column for pos tags
                           )

results = ranker.run(language_models, batch_size=64)
print(results)
```

#### For NER

```python3
ranker_ner = TransformerRanker(dataset_ner,
                               dataset_downsample=0.7,
                               text_column="tokens",
                               label_column="ner_tags",  # column for ner tags
                               )

results_ner = ranker_ner.run(language_models, batch_size=64)
print(results_ner)
```

Runtimes will vary depending on the dataset size and models used:

- For the **UD English Subtitles** dataset (downsampled to 2,621 texts), scoring 17 base-sized models took a total of 5 minutes—1.2 minutes to download the models and 3.8 minutes to embed and score them.
- For the **WNUT_17** dataset (downsampled to 3,977 texts), scoring the same 17 models took 8.6 minutes—1.2 minutes for model downloads and 7.4 minutes for embedding and scoring.

We used a GPU-enabled Colab Notebook (Tesla T4) for ranking.

## 4. Result Interpretation

After scoring, the results are sorted in descending order.
Transferability scores indicate each model's suitability for both datasets.

#### Results for PoS

```bash
Rank 1. FacebookAI/xlm-roberta-base: 11.6777
Rank 2. microsoft/mdeberta-v3-base: 11.6019
Rank 3. Twitter/twhin-bert-base: 11.5628
Rank 4. google/electra-base-discriminator: 11.4711
Rank 5. microsoft/deberta-v3-base: 11.4368
Rank 6. typeform/distilroberta-base-v2: 11.4277
Rank 7. bert-base-cased: 11.3257
Rank 8. distilbert-base-cased: 11.1967
Rank 9. roberta-base: 11.1803
Rank 10. Lianglab/PharmBERT-cased: 11.1535
Rank 11. dmis-lab/biobert-base-cased-v1.2: 11.1328
Rank 12. SpanBERT/spanbert-base-cased: 11.0394
Rank 13. sentence-transformers/all-mpnet-base-v2: 10.9533
Rank 14. german-nlp-group/electra-base-german-uncased: 10.8308
Rank 15. sentence-transformers/all-MiniLM-L12-v2: 10.3609
Rank 16. google/electra-small-discriminator: 10.3011
Rank 17. KISTI-AI/scideberta: 10.1395
```

#### For NER

```bash
Rank 1. microsoft/deberta-v3-base: 1.8297
Rank 2. sentence-transformers/all-mpnet-base-v2: 1.7781
Rank 3. roberta-base: 1.7535
Rank 4. typeform/distilroberta-base-v2: 1.6756
Rank 5. microsoft/mdeberta-v3-base: 1.6533
Rank 6. Twitter/twhin-bert-base: 1.6359
Rank 7. google/electra-base-discriminator: 1.6347
Rank 8. FacebookAI/xlm-roberta-base: 1.6175
Rank 9. bert-base-cased: 1.4237
Rank 10. sentence-transformers/all-MiniLM-L12-v2: 1.3134
Rank 11. german-nlp-group/electra-base-german-uncased: 1.276
Rank 12. Lianglab/PharmBERT-cased: 1.0724
Rank 13. distilbert-base-cased: 1.0338
Rank 14. google/electra-small-discriminator: 0.9056
Rank 15. KISTI-AI/scideberta: 0.8884
Rank 16. SpanBERT/spanbert-base-cased: 0.8103
Rank 17. dmis-lab/biobert-base-cased-v1.2: 0.7102
```

For CoNLL, the ranker points to _deberta-v3-base_ for fine-tuning, while for UD (PoS), it suggests using _xlm-roberta-base_.

## Summary

This markdown showed how to use `transformer-ranker` to choose the best-suited model for PoS tagging and NER datasets.
We loaded token classification datasets, selected language models, and ranked them based on transferability scores.
The result is a list of model names with transferability scores, helping you choose which to fine-tune or skip.

To fine-tune the top-ranked model, you can use Flair or Transformers frameworks.
