# Contributing to Open-Xtract

Thanks for your interest in contributing to Open-Xtract!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Mellow-Artificial-Intelligence/open-xtract.git
   cd open-xtract
   ```

2. Install dependencies with uv:
   ```bash
   uv sync --dev
   ```

3. Run tests:
   ```bash
   uv run pytest
   ```

4. Run linting:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   ```

## Making Changes

1. Create a new branch from `main`
2. Make your changes
3. Ensure tests pass and code is formatted
4. Submit a pull request

## Pull Request Guidelines

- PRs require approval from a code owner before merging
- Keep changes focused and atomic
- Update tests for new functionality
- Follow existing code style (enforced by ruff)

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run before submitting:

```bash
uv run ruff check . --fix
uv run ruff format .
```

## Questions?

Open an issue on GitHub.
