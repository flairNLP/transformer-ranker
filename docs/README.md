# Documentation

__Welcome__ to the docs (/tutorials) for TransformerRanker. It is a lightweight tool for ranking language models on downstream NLP tasks. 
It currently supports token, text, text-pair classification and regression tasks.

# Content

1. [Walkthrough](https://github.com/flairNLP/transformer-ranker/blob/main/docs/01-walkthrough.md): Introduces the tool with the first code example and a step-by-step guide.
2. [Learn by Example](https://github.com/flairNLP/transformer-ranker/blob/main/docs/02-examples.md): More examples on loading datasets and ranking models for NER, PoS, and text-pair classification.
3. [Advanced](https://github.com/flairNLP/transformer-ranker/blob/main/docs/03-advanced.md): Covers two optional parameters and introduces transferability metrics.

# Setup

This library is built as an add-on for transformers and relies on pytorch.
To use the latest version:

```bash
git clone https://github.com/FlairNLP/transformer-ranker
cd transformer-ranker
pip install -e .
```

This clones the repository and installs deps. 
After that, try running the [first code snippet](https://github.com/flairNLP/transformer-ranker/blob/main/docs/01-walkthrough.md#example-task) in the walkthrough.
