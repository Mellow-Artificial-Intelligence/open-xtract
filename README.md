<div align="center">

# OpenXtract

[![PyPI version](https://badge.fury.io/py/open-xtract.svg)](https://badge.fury.io/py/open-xtract)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/open-xtract)](https://pepy.tech/project/open-xtract)
[![GitHub stars](https://img.shields.io/github/stars/Mellow-Artificial-Intelligence/open-xtract.svg)](https://github.com/Mellow-Artificial-Intelligence/open-xtract/stargazers)

**Turn any document into structured data with AI**

*Open-source toolkit for extracting clean, structured data from text, images, and PDFs using large language models*

[Homepage](https://mellow-artificial-intelligence.github.io/open-xtract/) • [PyPI](https://pypi.org/project/open-xtract/) • [Documentation](https://github.com/Mellow-Artificial-Intelligence/open-xtract) • [Examples](./examples/)

</div>

---

## Quick Start

### Installation

```bash
# Using pip
pip install open-xtract

# Using uv (recommended)
uv add open-xtract
```

## Usage

### Input Types

The model string format: `<provider>:<model_string>`

**Examples**: `"openai:gpt-4o"`, `"anthropic:claude-3-5-sonnet-20241022"`, `"xai:grok-beta"`

```python
from pydantic import BaseModel
from open_xtract import OpenXtract

class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    vendor: str

ox = OpenXtract(model="openai:gpt-4o")

# Extract from text
result = ox.extract("Invoice #INV-2024-001 from TechCorp dated 2024-03-15 for $1,250.00", InvoiceData)

# Extract from image bytes
with open("receipt.png", "rb") as f:
    result = ox.extract(f.read(), InvoiceData)

# Extract from PDF bytes (automatically converts pages to images)
with open("invoice.pdf", "rb") as f:
    result = ox.extract(f.read(), InvoiceData)

print(result)
# InvoiceData(invoice_number='INV-2024-001', date='2024-03-15', total_amount=1250.0, vendor='TechCorp')
```

### Supported Models

```python
# OpenAI
ox = OpenXtract(model="openai:gpt-4o")
ox = OpenXtract(model="openai:gpt-4o-mini")

# Anthropic
ox = OpenXtract(model="anthropic:claude-3-5-sonnet-20241022")
ox = OpenXtract(model="anthropic:claude-3-5-haiku-20241022")

# Google
ox = OpenXtract(model="google:gemini-2.0-flash-exp")

# XAI
ox = OpenXtract(model="xai:grok-beta")

# OpenRouter (proxy to many models)
ox = OpenXtract(model="openrouter:qwen/qwen-2.5-72b-instruct")
```

### Complex Data Structures

```python
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class CompanyInfo(BaseModel):
    name: str
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

class DetailedInvoice(BaseModel):
    invoice_number: str
    date: datetime
    due_date: Optional[datetime] = None
    vendor: CompanyInfo
    customer: CompanyInfo
    line_items: List[LineItem]
    subtotal: float
    tax_amount: Optional[float] = None
    total_amount: float

# Extract complex nested structures
ox = OpenXtract(model="openai:gpt-4o")
result = ox.extract(complex_invoice_text, DetailedInvoice)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT - see [LICENSE](LICENSE).

---

Built by [Mellow AI](https://github.com/Mellow-Artificial-Intelligence)
