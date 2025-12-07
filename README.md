<div align="center">

# OpenXtract

[![PyPI version](https://badge.fury.io/py/open-xtract.svg)](https://badge.fury.io/py/open-xtract)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Extract structured data from any document using Claude**

Define a Pydantic schema, point to a file, get structured data back.

</div>

---

## How It Works

OpenXtract uses the [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/) to read documents and extract structured data. You define a Pydantic schema, and Claude extracts data matching that schema from text files, images, or PDFs.

## Installation

```bash
pip install open-xtract
```

## Usage

```python
import asyncio
from pydantic import BaseModel
from open_xtract import OpenXtract

class Invoice(BaseModel):
    invoice_number: str
    date: str
    total: float
    vendor: str

async def main():
    ox = OpenXtract()
    result = await ox.extract("invoice.pdf", Invoice)

    # Access extracted data
    print(result.data.invoice_number)
    print(result.data.total)

    # Access metadata
    print(f"Cost: ${result.cost_usd:.4f}")
    print(f"Tokens: {result.usage.total_tokens}")

asyncio.run(main())
```

## Extraction Result

`extract()` returns an `ExtractionResult[T]` with both the extracted data and metadata:

```python
result = await ox.extract("document.pdf", MySchema)

# Extracted data (your Pydantic model)
result.data

# Metadata
result.model          # Model used (e.g., "claude-sonnet-4-5-20250929")
result.cost_usd       # Cost in USD
result.duration_ms    # Total duration in milliseconds
result.num_turns      # Number of conversation turns
result.session_id     # Session identifier

# Token usage
result.usage.input_tokens
result.usage.output_tokens
result.usage.total_tokens
result.usage.cache_read_input_tokens
result.usage.cache_creation_input_tokens
```

## Configuration

```python
ox = OpenXtract(
    model="claude-sonnet-4-5",  # or "claude-opus-4-5", "claude-haiku-4-5"
    system_prompt="Extract dates in ISO 8601 format. Use full legal entity names.",
)
```

### Options

| Option | Description |
|--------|-------------|
| `model` | Claude model to use. Options: `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5` |
| `system_prompt` | Custom instructions for extraction behavior |
| `permission_mode` | Permission mode for tool execution. Default: `acceptEdits` |

## Supported File Types

| Type | Extensions |
|------|------------|
| Text | `.txt`, `.md`, `.json`, `.xml`, `.csv`, `.html`, `.yaml` |
| Images | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` |
| PDF | `.pdf` |

## Example: Academic Paper Extraction

```python
from pydantic import BaseModel, Field

class Author(BaseModel):
    name: str
    affiliation: str | None = None

class PaperMetadata(BaseModel):
    title: str = Field(description="The title of the paper")
    authors: list[Author] = Field(description="List of authors")
    abstract: str = Field(description="The paper abstract")
    arxiv_id: str | None = Field(default=None, description="arXiv identifier")
    keywords: list[str] = Field(default_factory=list)

async def main():
    ox = OpenXtract(
        model="claude-sonnet-4-5",
        system_prompt="Extract academic paper metadata accurately.",
    )

    result = await ox.extract("paper.pdf", PaperMetadata)

    print(f"Title: {result.data.title}")
    print(f"Authors: {[a.name for a in result.data.authors]}")
    print(f"Cost: ${result.cost_usd:.4f}")
```

## License

MIT
