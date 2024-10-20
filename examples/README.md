# Examples

This directory provides examples for loading different datasets, choosing language models, and ranking them.
The library has two main goals: to rank language models quickly and effectively, and to make the repository easy to use, promoting less fine-tuning.
Most examples are short, just 6 lines of code, but come with detailed explanations:

1. [Rank Language Models for Text Classification](https://github.com/flairNLP/transformer-ranker/blob/main/examples/01-text-classification.md)
2. [Rank Language Models for Sequence Labeling](https://github.com/flairNLP/transformer-ranker/blob/main/examples/02-sequence-labeling.md)
3. [Bonus: Understanding Estimators](https://github.com/flairNLP/transformer-ranker/blob/main/examples/03-advanced.md#transferability-estimation)
4. [Bonus: Layerwise Analysis](https://github.com/flairNLP/transformer-ranker/blob/main/examples/03-advanced.md#layerwise-analysis)

# Quick Summary

This library works as an add-on for [Transformers](https://github.com/huggingface/transformers) and relies on PyTorch.
The only requirements are those two libraries, assuming you’ve already set up your Python environment.
To use the latest version, clone the repository and install the dependencies:

```bash
git clone https://github.com/FlairNLP/transformer-ranker
cd transformer_ranker
pip install -r requirements.txt
```

All code snippets are listed below. Copy and tweak them as needed. _Click to expand_ 👻

<details>
<summary>
Text classification with 'trec'<br>
</summary>

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load dataset
dataset = load_dataset('trec')

# Load a list of popular base-sized models
language_models = prepare_popular_models('base')

# Initialize the ranker
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# ... and run it
results = ranker.run(language_models, batch_size=64)

# Inspect results
print(results)
```
</details>

<details>
<summary>
NER with 'conll2003'<br>
</summary>

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load dataset
dataset = load_dataset('conll2003')

# Load a list of popular base-sized models
language_models = prepare_popular_models('base')

# Initialize the ranker
ranker = TransformerRanker(dataset,
                           dataset_downsample=0.2,
                           text_column='tokens',
                           label_column='ner_tags')

# ... and run it
results = ranker.run(language_models, batch_size=64)

# Inspect results
print(results)
```
</details>

<details>
<summary>
PoS tagging with UD's 'en_lines'<br>
</summary>

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Load the collection of Universal Dependencies and specify the dataset
dataset = load_dataset('universal-dependencies/universal_dependencies', 'en_lines')

# Load a list of popular base-sized models
language_models = prepare_popular_models('base')

# Initialize the ranker
ranker = TransformerRanker(dataset,
                           dataset_downsample=0.5,
                           text_column="tokens",
                           label_column="upos")

# ... and run it
results = ranker.run(language_models, batch_size=64)

# Inspect results
print(results)
```
</details>


<details>
<summary>
Text pair classification with 'rte'<br>
</summary>

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load the 'SetFit/rte' dataset for the Recognizing Textual Entailment
dataset = load_dataset('SetFit/rte')

# Prepare some language models to choose from 
language_models = ['prajjwal1/bert-tiny',
                   'google/electra-small-discriminator',
                   'microsoft/deberta-v3-small',
                   'bert-base-uncased',
                   ]

# Initialize the ranker with the dataset and text columns
ranker = TransformerRanker(dataset=dataset,
                           dataset_downsample=0.5,
                           text_column='text1',
                           text_pair_column='text2',  # add text pair column for entailment-type tasks
                           label_column='label',
                           )

# ... and run it
result = ranker.run(models=language_models, batch_size=32)

# Inspect results
print(result)
```
</details>


<details>
<summary>
Text pair regression with 'stsb'<br>
</summary>

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load the 'SetFit/stsb' dataset used for the Semantic Textual Similarity Benchmark (STS-B)
dataset = load_dataset('SetFit/stsb')

# Define language models
models = ['prajjwal1/bert-tiny',
          'google/electra-small-discriminator',
          'microsoft/deberta-v3-small',
          'bert-base-uncased',
          ]

# Initialize the ranker
ranker = TransformerRanker(dataset=dataset,
                           text_column='text1',
                           text_pair_column='text2',
                           label_column='label',  # label column that holds floats
                           )

# ... and run it with logme estimator
result = ranker.run(models=models,
                    estimator="logme",  # use logme for regression tasks
                    batch_size=32,
                    )

# Inspect results
print(result)
```
</details>

<details>
<summary>
Best layer search in deberta-v2-xxlarge<br>
</summary>

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Load the CoNLL dataset
dataset = load_dataset('conll2003')

# Let's use a single language model
language_model = ['microsoft/deberta-v2-xxlarge']

# Initialize the ranker and downsample the dataset
ranker = TransformerRanker(dataset=dataset, dataset_downsample=0.2)

# Run it using the 'bestlayer' option
result = ranker.run(models=language_model, layer_aggregator='bestlayer')

# Inspect scores for each layer
print(result)
```

Layer -41 (7th from the embedding layer) has the highest transferability score.
You can find the scores for all layers in the layerwise scores dictionary.

```bash
INFO:transformer_ranker.ranker:microsoft/deberta-v2-xxlarge, score: 2.8912 (layer -41)
layerwise scores: {-1: 2.7377, -2: 2.8024, -3: 2.8312, -4: 2.8270, -5: 2.8293, -6: 2.7952, -7: 2.7894, -8: 2.7777, -9: 2.7490, -10: 2.7020, -11: 2.6537, -12: 2.7227, -13: 2.6930, -14: 2.7187, -15: 2.7494, -16: 2.7002, -17: 2.6834, -18: 2.6210, -19: 2.6126, -20: 2.6459, -21: 2.6693, -22: 2.6730, -23: 2.6475, -24: 2.7037, -25: 2.6768, -26: 2.6912, -27: 2.7300, -28: 2.7525, -29: 2.7691, -30: 2.7436, -31: 2.7702, -32: 2.7866, -33: 2.7737, -34: 2.7550, -35: 2.7269, -36: 2.7723, -37: 2.7586, -38: 2.7969, -39: 2.8551, -40: 2.8692, -41: 2.8912, -42: 2.8530, -43: 2.8646, -44: 2.8655, -45: 2.8210, -46: 2.7836, -47: 2.6945, -48: 2.5153}
```
</details>


## Important notes

__Inspect the dataset__ Check the dataset before using it in the ranker. Make sure it has both texts and labels. 

__Dataset downsampling__ Set the `dataset_downsample` ratio based on the dataset size. Try not to downsample too much, especially if you're working with small datasets that have lots of classes. It’s recommended to have at least 1000 text instances.

__Rankings__ The final result displays language model names and their transferability scores in descending order. These scores are calculated using transferability metrics, which estimate each model's suitability for your dataset.

# Additional Resources

If curious what the ranking scores represent, take a look at the advanced tutorial: _How is the suitability of language models estimated?_
Next up: a notebooks folder with a hands-on Colab example to rank LMs and fine-tune the highest-ranked one using Flair... and Spaces integration.
