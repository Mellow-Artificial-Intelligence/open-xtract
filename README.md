# openextract

Extract structured data from documents, images, audio, and video using LLMs.

## Installation

```bash
uv add openextract
```

Or

```bash
pip install openextract
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
    model="openai:gpt-5.4",
    input_file="https://example.com/document.pdf",
    instructions="return a 2 sentence summary and the primary language of the document",
)
print(result)
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
