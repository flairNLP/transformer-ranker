# Contributing to TransformerRanker

__Thanks__ for your interest in contributing to `transformer-ranker`! We're building a lightweight tool for ranking language models via transferability.
Finding bugs, proposing enhancements, or opening pull requests, your involvement shapes the project.

## Before Coding

Please check existing [issues](https://github.com/flairNLP/transformer-ranker/issues) and [pull requests](https://github.com/flairNLP/transformer-ranker/pulls) to avoid duplicate work.

If you're working on something new, consider opening an issue first to discuss it.

## Setup

TransformerRanker requires Python 3.9 or higher.  
We recommend using Python 3.9.x during development.

Install dev dependencies with:

```bash
pip install -e ".[dev]"
```

Make sure you’re using a clean environment (conda or virtualenv).  
Check out the examples and tests to understand the interface better. 

## Formatting

We use [`black`](https://github.com/psf/black) for formatting [`ruff`](https://github.com/charliermarsh/ruff) for linting and imports.

To auto-format your code:

```bash
black . && ruff --fix .
```

## Testing

We use pytest for running unit tests. Run all tests using:

```bash
pytest tests
```

## Adding Transferability Metrics

New metrics should be added to `transformer-ranker/estimators/`. When contributing, include docstrings with links to relevant papers to help others understand the approach. Use comments to make your code easy to follow.

## Enhancements

Test the tool with unusual models and datasets. Open issues or PRs if you encounter unsupported models. Consider exploring ways to apply transferability to other tasks, such as summarization.

## Get Involved

Ways to contribute:

- Open issues to report bugs, suggest improvements, or ask questions.
- Add new transferability metrics.
- Test the tool with unusual models and datasets.

Hope this helps, we actively welcome your contributions!
