# Contributing to TransformerRanker

__Thanks__ for your interest in contributing to `transformer-ranker`. Whether it’s a bug report, new feature, or doc fix, it’s appreciated. Please read this before opening an issue or PR.

## Reporting Bugs / Requesting Features

Use [issues](https://github.com/flairNLP/transformer-ranker/issues) to report bugs or suggest improvements.

Before opening a new issue:

- See if similar issues and PRs already exist
- If not, feel free to create a new one

Helpful info to include:

- What you did and what went wrong
- Reproducible steps or a minimal example (if possible)

## Contributing with Pull Requests

Pull requests are welcome.

Before submitting:

- Make sure you're on top of `main`
- For larger changes, open an issue first to discuss

Steps:

1. Fork the repo and create a new branch
2. Make your changes (keep it focused)
3. Format and test
4. Commit with a clear message
5. Open a PR and respond to feedback if needed

## Dev Setup

Use Python 3.9.x in a clean environment (e.g., virtualenv or conda).

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

We use:

- [black](https://github.com/psf/black) — code formatting
- [ruff](https://github.com/astral-sh/ruff) — linting & import sorting
- [pytest](https://docs.pytest.org/) — testing

To format and lint:

```bash
black . && ruff --fix .
```

To run tests:

```bash
pytest tests
```

---

## Adding New Metrics

New estimators go in:

```
transformer_ranker/estimators/
```

Please include:

- Docstrings and helpful comments
- Paper references (if relevant)
- Associated tests in `tests/` (if possible)

## Enhancements

Test the tool with new models and datasets. Open issues or PRs if you encounter anything unsupported. Consider exploring ways to apply transferability to other tasks, such as summarization.

## License

MIT. See [LICENSE](./LICENSE). By contributing, you agree your code is shared under the same license.
