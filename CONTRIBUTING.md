# Contributing to openextract

Thanks for your interest in contributing to openextract!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Mellow-Artificial-Intelligence/openextract.git
   cd openextract
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

## Dependency embargo (24h)

We do **not** install any package version that has been published less than 24 hours ago. This protects against supply-chain attacks where a compromised release sits on PyPI for a few hours before being yanked.

The policy is enforced via uv's `exclude-newer` setting in `pyproject.toml`:

```toml
[tool.uv]
exclude-newer = "<YYYY-MM-DDTHH:MM:SSZ>"
```

A scheduled GitHub Actions workflow (`.github/workflows/embargo-bump.yml`) runs daily at 03:00 UTC and opens a PR that advances the cutoff to "yesterday 00:00 UTC". Merge those PRs as part of normal review. The workflow uses only first-party tooling (the `gh` CLI preinstalled on GitHub-hosted runners and `astral-sh/setup-uv`) — no third-party PR-creation actions, to keep the supply-chain surface for the embargo workflow itself minimal.

If you need to add a dependency that was published in the last 24 hours, hold off and wait the embargo out rather than bumping `exclude-newer` past the rolling window.

## Questions?

Open an issue on GitHub.
