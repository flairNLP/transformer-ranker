# Documentation

Welcome to the documentation (/tutorials) of the TransformerRanker. This tool lets you rank language models LMs for downstream NLP tasks. 
It currently supports token, text, and text-pair classification, as well as regression tasks.

# Content

1. [Walkthrough](https://github.com/flairNLP/transformer-ranker/blob/main/docs/01-walkthrough.md): introduces the tool with a first code example and detailed step-by-step guide.
2. [Learn by example](https://github.com/flairNLP/transformer-ranker/blob/main/examples/02-examples.md): offers additional examples on loading datasets and ranking models for NER, PoS, and text-pair classification tasks.
3. [Advanced](https://github.com/flairNLP/transformer-ranker/blob/main/examples/03-advanced.md): details two optional parameters and introduces transferability metrics.

# Setup

This library is built as an add-on for transformers and relies on pytorch.
To use the latest version:

```bash
git clone https://github.com/FlairNLP/transformer-ranker
cd transformer-ranker
pip install -e .
```

This clones the repository and installs deps. 
Then, you can start by running the [first code example](https://github.com/flairNLP/transformer-ranker/blob/main/docs/01-walkthrough.md#example-task) from the walkthrough.
