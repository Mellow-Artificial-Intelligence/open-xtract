# OpenXtract

**Turn documents into structured data**

Open-source toolkit for extracting clean, structured data from text.

- [GitHub](https://github.com/Mellow-Artificial-Intelligence/open-xtract)
- [PyPI](https://pypi.org/project/open-xtract/)

## Installation

```bash
pip install open-xtract
# or
uv add open-xtract
```

## Usage

```python
from pydantic import BaseModel
from open_xtract import OpenXtract

class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    vendor: str

ox = OpenXtract(model="gpt-4o")  # or any OpenAI model

# Extract from text
result = ox.extract("Total: $123.45 on 2025-03-01 from ACME", InvoiceData)
print(result)
```

## Advanced Features

### Model Configuration

```python
# Use any OpenAI-compatible model
ox = OpenXtract(model="gpt-4o", api_key="your-key")
ox = OpenXtract(model="grok-4", base_url="https://api.x.ai/v1", api_key="your-xai-key")
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
