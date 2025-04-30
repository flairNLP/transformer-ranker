# Contributing to TransformerRanker

__Thanks__ for your interest in contributing to `transformer-ranker`! We're building a lightweight tool for ranking language models via transferability. 

Spotted a bug, got an idea, or want to open a PR? any kind of involvement helps shape where it goes.

## See Exiting Work

Before starting, check open [issues](https://github.com/flairNLP/transformer-ranker/issues) and [pull requests](https://github.com/flairNLP/transformer-ranker/pulls) to avoid duplicates.
Planning something new? Consider opening an issue first. Discussion is welcome.

## Setup

We recommend Python 3.9.x during development. To install dev dependencies:

```bash
pip install -e ".[dev]"
```

Use a clean environment (virtualenv or conda).
Explore `examples/` and `tests/` to get familiar.

## Code Style & Linting

We use:

- [black](https://github.com/psf/black) — code formatting
- [ruff](https://github.com/astral-sh/ruff) — linting & import sorting

To format:

```bash
black . && ruff --fix .
```

## Run Tests

Tests are written using pytest. Run all tests with:

```bash
pytest tests
```

## Add New Transferability Metrics


New metrics go in `transformer_ranker/estimators/`.
Include docstrings and paper references. Clear comments = appreciated.

## Explore and Expand

Try weird models or datasets. If something fails — open an issue.
Ideas for model ranking for tasks like summarization are also welcome.

## Enhancements

Test the tool with unusual models and datasets. Open issues or PRs if you encounter unsupported models. Consider exploring ways to apply transferability to other tasks, such as summarization.

----------------

Contributions big or small are appreciated! 🚀
Want to suggest something now? → Open an [issues](https://github.com/flairNLP/transformer-ranker/issues) 
