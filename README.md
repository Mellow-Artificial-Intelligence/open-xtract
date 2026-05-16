<div align="center">

# openextract

**Extract structured data from documents, images, audio, and video using LLMs.**

[![PyPI version](https://img.shields.io/pypi/v/openextract.svg?logo=pypi&logoColor=white&color=4B8BBE)](https://pypi.org/project/openextract/)
[![Python versions](https://img.shields.io/pypi/pyversions/openextract.svg?logo=python&logoColor=white)](https://pypi.org/project/openextract/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://img.shields.io/pypi/dm/openextract.svg?color=blue)](https://pypi.org/project/openextract/)

[Documentation](https://mellow-artificial-intelligence.github.io/openextract/) &middot; [PyPI](https://pypi.org/project/openextract/) &middot; [Changelog](CHANGELOG.md) &middot; [Issues](https://github.com/Mellow-Artificial-Intelligence/openextract/issues)

</div>

---

`openextract` turns any document, image, audio, or video file into a typed Pydantic model in a single function call. Point it at a local path or a URL, pass a schema, and get back a validated object you can use directly in your code.

## Features

- **Type-safe output.** Define your shape with Pydantic; get back a validated instance.
- **One function, many modalities.** Documents (PDF, DOCX), images, audio, and video.
- **Local files or URLs.** Pass a path or an `https://` URL &mdash; `openextract` handles fetching.
- **Bring your own model.** OpenAI, Google, and Ollama supported out of the box via [`pydantic-ai`](https://github.com/pydantic/pydantic-ai).
- **Explicit error handling.** Distinct exceptions for URL fetch, schema validation, and model errors.
- **100% test coverage**, enforced in CI.

## Installation

```bash
uv add openextract
```

Or with pip:

```bash
pip install openextract
```

Requires Python 3.12+.

## Quick start

```python
from pydantic import BaseModel
from openextract import extract


class PdfInfo(BaseModel):
    summary: str
    language: str


result = extract(
    schema=PdfInfo,
    model="openai:gpt-5",
    input_file="https://example.com/document.pdf",
    instructions="Return a two-sentence summary and the document's primary language.",
)

print(result.summary)
print(result.language)
```

`result` is a fully-validated `PdfInfo` instance &mdash; not a dict, not a string.

## Usage

### Local files

```python
result = extract(
    schema=PdfInfo,
    model="openai:gpt-5",
    input_file="./reports/q4.pdf",
)
```

### Choosing a model

`model` follows the `pydantic-ai` provider prefix convention:

| Provider | Example identifier         |
| -------- | -------------------------- |
| OpenAI   | `openai:gpt-5`             |
| Google   | `google-gla:gemini-2.5-pro`|
| Ollama   | `ollama:llama3`            |

Set the corresponding provider credentials in your environment (e.g. `OPENAI_API_KEY`). `openextract` loads `.env` automatically.

### Error handling

```python
from openextract import (
    extract,
    UrlFetchError,
    SchemaValidationError,
    ModelError,
    ExtractionError,
)

try:
    result = extract(schema=PdfInfo, model="openai:gpt-5", input_file=url)
except UrlFetchError:
    ...  # The URL could not be fetched
except SchemaValidationError:
    ...  # The model's output did not match your schema
except ModelError:
    ...  # The model provider returned an error
except ExtractionError:
    ...  # Any other extraction failure (base class)
```

All `openextract` exceptions inherit from `ExtractionError`, so you can catch it as a single fallback if you prefer.

## API reference

### `extract(schema, model, input_file, instructions=None)`

| Argument       | Type              | Description                                                      |
| -------------- | ----------------- | ---------------------------------------------------------------- |
| `schema`       | `type[BaseModel]` | A Pydantic model class describing the desired output shape.      |
| `model`        | `str`             | A `pydantic-ai` model identifier (e.g. `"openai:gpt-5"`).        |
| `input_file`   | `str`             | A local file path or an `https://` URL.                          |
| `instructions` | `str \| None`     | Optional natural-language guidance for the model.                |

Returns an instance of `schema`.

## Development

```bash
git clone https://github.com/Mellow-Artificial-Intelligence/openextract.git
cd openextract
uv sync --dev

uv run pytest --cov=openextract            # tests + coverage
uv run ruff check .                        # lint
uv run ruff format --check .               # format check
```

CI runs the test suite on every PR and fails if total coverage drops below 100%.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

## License

[MIT](LICENSE) &copy; Cole McIntosh
