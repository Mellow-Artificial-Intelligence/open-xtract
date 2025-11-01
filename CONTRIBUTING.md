# Contributing to open-xtract

Thank you for your interest in contributing to open-xtract!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Mellow-Artificial-Intelligence/open-xtract.git
cd open-xtract
```

2. Install in development mode with all dependencies:
```bash
uv sync
```

## Running Tests

```bash
pytest tests/
```

## Making Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your fork (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Code Style

- Use type hints for all function parameters and returns
- Add docstrings to all public functions and classes
- Follow PEP 8 style guidelines

## Areas for Contribution

- Additional LLM provider implementations
- New document types beyond PDF
- Performance optimizations
- Documentation improvements
- Test coverage expansion
- Bug fixes

## Questions?

Feel free to open an issue for any questions or discussions!

## Releasing to PyPI

Releases are automated via GitHub Actions. To release a new version:

### Prerequisites

1. **Set up PyPI trusted publishing** (recommended):
   - Go to [PyPI Account Settings](https://pypi.org/manage/account/)
   - Navigate to **API tokens** → **Add API token** → **Trusted publisher**
   - Add your GitHub repository as a trusted publisher
   - The workflow will automatically authenticate using GitHub's OIDC

2. **OR use API token** (alternative):
   - Create an API token on PyPI
   - Add it as a GitHub secret named `PYPI_API_TOKEN`
   - Update `.github/workflows/release.yml` to use the token instead of trusted publishing

### Release Process

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.3"  # Update to new version
   ```

2. **Commit and push** the version change:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.3"
   git push
   ```

3. **Create a GitHub release**:
   - Go to the repository's Releases page
   - Click "Create a new release"
   - Tag: `v0.1.3` (must match the version in pyproject.toml, prefixed with 'v')
   - Title: `Release v0.1.3` (or your release title)
   - Description: Add release notes
   - Click "Publish release"

4. **Automated release**: The GitHub Action will:
   - Build the package
   - Verify version matches
   - Publish to PyPI automatically

**Alternative**: You can also trigger the workflow manually via "Actions" → "Release to PyPI" → "Run workflow" (it will use the version from pyproject.toml).
