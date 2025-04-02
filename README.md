<p align="center">A very simple library that helps you find the <b>best-suited language model</b> for your NLP task.
Developed at <a href="https://www.informatik.hu-berlin.de/en/forschung-en/gebiete/ml-en/">Humboldt University of Berlin</a>.
</p>
<p align="center">
<a href="https://pypi.org/project/transformer-ranker/"><img alt="PyPi version" src="https://badge.fury.io/py/transformer-ranker.svg"></a>
<img alt="python" src="https://img.shields.io/badge/python-3.9-blue">
<img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-green">
<a href="https://huggingface.co/spaces/lukasgarbas/transformer-ranker"><img alt="Demo Spaces" src="https://img.shields.io/badge/Demo-Spaces-brightgreen"></a>
</p>
<div align="center">
<hr>

[Quick Start](#quick-start) | [Tutorials](#tutorials) | [Demonstration Paper](https://arxiv.org/abs/2409.05997) | [Approach Paper](https://aclanthology.org/2024.findings-acl.757/)

</div>


---
**The problem**: There are too many pre-trained language models (LMs) out there.
But which one of them is best for your NLP classification task? 
Since fine-tuning LMs is costly, it is not possible to try them all!  

**The solution**: *Transferability estimation* with TransformerRanker!

---
TransformerRanker is a library that

* **quickly finds the best-suited language model for a given NLP classification task.** 
  All you need to do is to select a [dataset](https://huggingface.co/datasets) and a list of pre-trained [language models](https://huggingface.co/models) (LMs) from the 🤗 HuggingFace Hub. TransformerRanker will quickly estimate which of these LMs will perform best on the given task!

* **efficiently performs layerwise analysis of LMs.** Transformer LMs have many layers. Use TransformerRanker to identify which intermediate layer
  is best-suited for a downstream task!

<hr> 

## Quick Start

To install from pip, simply do:

```python3
pip install transformer-ranker
```

## Example 1: Find the best LM for Named Entity Recognition 

Let's say we want to find the best LM for English Named Entity Recognition (NER) on the popular CoNLL-03 dataset. 

To keep this example simple, we use TransformerRanker to only choose between two models: `bert-base-cased` and `bert-base-uncased`. 

The full snippet to do so is as follows: 

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker

# Step 1: Load the CoNLL-03 dataset from HuggingFace
dataset = load_dataset('conll2003')

# Step 2: Define the LMs to choose from 
language_models = ["bert-base-cased", "bert-base-uncased"]

# Step 3: Initialize the ranker with the dataset 
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# ... and run the ranker to obtain the ranking
results = ranker.run(language_models, batch_size=64)
```

If you run this snippet for the first time, it will first download the CoNLL-03 dataset from HuggingFace, and also 
download the two transformer LMs. It will then conduct the estimation for the two LMs. On a GPU-enabled Google Colab 
notebook, this should only take a minute or two. 

Print the results by doing

```python3
print(results)
```

This should print: 

```console
Rank 1. bert-base-uncased: 2.5935
Rank 2. bert-base-cased: 2.5137
```

This indicates that the uncased variant of BERT is likely to perform better on CoNLL-03!


## Example 2: Really find the best LM 

The first example was kept simple: we only chose between two LMs. But in practical use cases, you might want to
choose between **dozens** of LMs. 

To help you get started, we compiled two lists of popular LMs that in our opinion are good LMs to try:
1. A 'base' list that contains 17 popular models of medium size.
2. A 'large' list that contains popular models of larger size.
   
To find the best LM for English NER among 17 base LMs, use the following snippet:

```python3
from datasets import load_dataset
from transformer_ranker import TransformerRanker, prepare_popular_models

# Step 1: Load the CoNLL-03 dataset from HuggingFace
dataset = load_dataset('conll2003')

# Step 2: Use our list of 17 'base' LMs as candidates 
language_models = prepare_popular_models('base')

# Step 3: Initialize the ranker with the dataset 
ranker = TransformerRanker(dataset, dataset_downsample=0.2)

# ... and run the ranker to obtain the ranking
results = ranker.run(language_models, batch_size=64)

# print the ranking
print(results)
```

Done! This will print: 

```console
Rank 1. microsoft/deberta-v3-base: 2.6739
Rank 2. google/electra-base-discriminator: 2.6115
Rank 3. microsoft/mdeberta-v3-base: 2.6099
Rank 4. roberta-base: 2.5919
Rank 5. typeform/distilroberta-base-v2: 2.5834
Rank 6. sentence-transformers/all-mpnet-base-v2: 2.5709
Rank 7. bert-base-cased: 2.5137
Rank 8. FacebookAI/xlm-roberta-base: 2.4894
Rank 9. Twitter/twhin-bert-base: 2.4261
Rank 10. german-nlp-group/electra-base-german-uncased: 2.2517
Rank 11. distilbert-base-cased: 2.1989
Rank 12. sentence-transformers/all-MiniLM-L12-v2: 2.1957
Rank 13. Lianglab/PharmBERT-cased: 2.1945
Rank 14. google/electra-small-discriminator: 1.945
Rank 15. KISTI-AI/scideberta: 1.9175
Rank 16. SpanBERT/spanbert-base-cased: 1.7301
Rank 17. dmis-lab/biobert-base-cased-v1.2: 1.5784
```

This ranking gives you an indication which models might perform best on CoNLL-03.
Accordingly, you can exclude the lower-ranked models and focus on the top-ranked models.

*Note:* Doing estimation for all 17 base models will take about 15 minutes on a GPU-enabled Colab Notebook (most time is spent 
downloading the models if you don't already have them locally). 



## Tutorials

We provide **tutorials** to introduce the library and key concepts:

1. [**Tutorial 1: Library Walkthrough**](docs/01-walkthrough.md)
2. [**Tutorial 2: Learn by Example**](docs/02-examples.md)
3. [**Tutorial 3: Advanced**](docs/03-advanced.md)

## Cite

Please cite the following [paper](https://arxiv.org/abs/2409.05997) when using TransformerRanker or building upon our work:

```bibtex
@misc{garbas2024transformerrankertoolefficientlyfinding,
      title={TransformerRanker: A Tool for Efficiently Finding the Best-Suited Language Models for Downstream Classification Tasks}, 
      author={Lukas Garbas and Max Ploner and Alan Akbik},
      year={2024},
      eprint={2409.05997},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.05997}, 
}
```

## Contact

Please email your questions or comments to [**Lukas Garbas**](mailto:lukas.garbaciauskas@informatik.hu-berlin.de?subject=[GitHub]%20TransformerRanker)

## Contributing

Thanks for your interest in contributing! There are many ways to get involved;
check these [open issues](https://github.com/flairNLP/transformer-ranker/issues) for specific tasks.

## License

[MIT](LICENSE)
