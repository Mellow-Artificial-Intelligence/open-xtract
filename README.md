# openextract

[![PyPI version](https://img.shields.io/pypi/v/openextract.svg)](https://pypi.org/project/openextract/)
[![PyPI downloads](https://img.shields.io/pypi/dm/openextract.svg)](https://pypi.org/project/openextract/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml/badge.svg)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/Mellow-Artificial-Intelligence/openextract)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![pydantic-ai](https://img.shields.io/badge/pydantic--ai-1.37+-7C3AED.svg)](https://ai.pydantic.dev/)

Extract structured data from documents, images, audio, and video using LLMs.

## Installation

```bash
uv add openextract
```

## Usage

```python
from pydantic import BaseModel
from openextract import extract

class PdfInfo(BaseModel):
    summary: str
    language: str

result = extract(
    schema=PdfInfo,
    model="openai-responses:gpt-5.2",
    url="https://example.com/document.pdf",
    instructions="return a 2 sentence summary and the primary language of the document",
)
print(result)
```

## Logging

To enable logfire instrumentation for tracing:

```python
from openextract import configure_logging

configure_logging()
```

## Error Handling

```python
from openextract import (
    extract,
    ExtractionError,
    ModelError,
    SchemaValidationError,
    UrlFetchError,
)

try:
    result = extract(...)
except UrlFetchError as e:
    print(f"Failed to fetch URL: {e}")
except SchemaValidationError as e:
    print(f"Output didn't match schema: {e}")
except ModelError as e:
    print(f"Model API error: {e}")
except ExtractionError as e:
    print(f"Extraction failed: {e}")
```

## Supported Media Types

| Type | Extensions |
|------|------------|
| Documents | `.pdf`, `.doc`, `.docx`, `.txt`, `.html`, `.csv`, `.xls`, `.xlsx` |
| Images | `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.svg` |
| Audio | `.mp3`, `.wav`, `.ogg`, `.flac`, `.aac`, `.m4a` |
| Video | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.wmv` |

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
