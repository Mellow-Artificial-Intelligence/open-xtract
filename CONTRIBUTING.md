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
pip install -e ".[vision,dev]"
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
