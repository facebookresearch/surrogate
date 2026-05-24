# Contributing to surrogate

Thanks for your interest in contributing! This document outlines how to
propose changes and report issues.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository and clone your fork locally.
2. Create a new branch for your work:
   ```bash
   git checkout -b my-feature
   ```
3. Set up the development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development Workflow

### Running Tests

Run the full test suite before submitting changes:

```bash
pytest tests/
```

To run a single test file:

```bash
pytest tests/test_surrogate_model.py
```

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use clear, descriptive names for variables, functions, and classes.
- Add docstrings to public functions and classes.
- Keep changes focused — avoid mixing refactors with feature work.

## Reporting Issues

When opening an issue, please include:

- A clear, descriptive title.
- Steps to reproduce the problem.
- Expected vs. actual behavior.
- Your environment (Python version, OS, relevant package versions).
- A minimal code example, if applicable.

## Submitting Changes

1. Ensure your branch is up to date with the main branch.
2. Make sure all tests pass and new code has appropriate test coverage.
3. Write a clear commit message describing the change and motivation.
4. Push your branch and open a pull request.
5. Reference any related issues in the pull request description.

### Pull Request Checklist

- [ ] Tests added or updated for the change.
- [ ] All tests pass locally.
- [ ] Code follows the project's style conventions.
- [ ] Documentation updated (README, docstrings) where relevant.
- [ ] Commit messages are clear and descriptive.

## Adding New Features

For larger features, please open an issue first to discuss the proposed
change. This avoids duplicated work and helps align on the design.

## Questions

If you have questions about contributing, feel free to open an issue with the
`question` label.
