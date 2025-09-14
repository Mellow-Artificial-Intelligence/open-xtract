# OpenXtract

**Turn documents into structured data**

Open-source toolkit for extracting clean, structured data from text, images, and PDFs.

- [GitHub](https://github.com/Mellow-Artificial-Intelligence/open-xtract)
- [PyPI](https://pypi.org/project/open-xtract/)

## Installation

```bash
pip install open-xtract
# or
uv add open-xtract
```

## Usage

The model string should look like: `<provider>:<model_string>`

Ex. "openai:gpt-5-nano", "xai:grok-4"

```python
from pydantic import BaseModel
from open_xtract import OpenXtract

class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    vendor: str

ox = OpenXtract(model="openai:gpt-5-nano")  # or any model

# Extract from text (str)
result = ox.extract("Total: $123.45 on 2025-03-01 from ACME", InvoiceData)
print(result)

# Extract from image (bytes)
with open("/path/to/receipt.png", "rb") as f:
    img_bytes = f.read()
result = ox.extract(img_bytes, InvoiceData)
print(result)

# Extract from PDF (bytes) — each page is rendered to an image internally
with open("/path/to/invoice.pdf", "rb") as f:
    pdf_bytes = f.read()
result = ox.extract(pdf_bytes, InvoiceData)
print(result)
```

## Advanced Features

### Model Configuration

```python
# Use any OpenAI-compatible model
ox = OpenXtract(model="openrouter:qwen/qwen3-max")
ox = OpenXtract(model="xai:grok-4")
```

## Features

- Extract structured data from text
- Model-agnostic (works with any OpenAI-compatible API)
- Simple, clean API

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT - see [LICENSE](LICENSE).

---

Built with ❤️ by [Mellow AI](https://github.com/Mellow-Artificial-Intelligence)
