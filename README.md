<div align="center">

# OpenXtract

[![PyPI version](https://badge.fury.io/py/open-xtract.svg)](https://badge.fury.io/py/open-xtract)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/open-xtract)](https://pepy.tech/project/open-xtract)
[![GitHub stars](https://img.shields.io/github/stars/Mellow-Artificial-Intelligence/open-xtract.svg)](https://github.com/Mellow-Artificial-Intelligence/open-xtract/stargazers)

**Turn any document into structured data with AI**

*Open-source toolkit for extracting clean, structured data from text, images, and PDFs using state-of-the-art large language models*

[Homepage](https://mellow-artificial-intelligence.github.io/open-xtract/) • [PyPI](https://pypi.org/project/open-xtract/) • [Documentation](https://github.com/Mellow-Artificial-Intelligence/open-xtract)

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

### Basic Usage

```python
from pydantic import BaseModel
from open_xtract import OpenXtract

# Define your data structure
class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    vendor: str

# Initialize extractor
ox = OpenXtract(model="openai:gpt-4o")

# Extract from any input type
result = ox.extract("Invoice #123 from ACME Corp on 2025-03-01 for $456.78", InvoiceData)
print(result)
# InvoiceData(invoice_number='123', date='2025-03-01', total_amount=456.78, vendor='ACME Corp')
```

## Features

- **Universal Input Support**: Extract from text, images (PNG, JPG), and PDFs
- **Model Agnostic**: Works with OpenAI, Anthropic, Google, XAI, and any OpenAI-compatible API
- **Type-Safe**: Built on Pydantic for guaranteed data structure validation
- **Fast & Efficient**: Optimized extraction pipeline with minimal overhead
- **Precise**: Advanced prompt engineering for accurate structured data extraction
- **Simple API**: One method to extract from any input type

## Detailed Usage

### Input Types

The model can be specified in two formats:

1. **With colon**: `<provider>:<model_string>` (e.g., `"openai:gpt-4o"`)
2. **Without colon**: `<model_string>` when `provider` parameter is provided separately

**Examples**:
- `OpenXtract(model="openai:gpt-4o")`
- `OpenXtract(model="gpt-4o", provider="openai")`
- `OpenXtract(model="anthropic:claude-3-5-sonnet-20241022")`
- `OpenXtract(model="xai:grok-beta")`

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

### Configuration Options

You can configure OpenXtract using environment variables (default) or by passing parameters directly:

```python
# Using environment variables (default)
# Set OPENAI_API_KEY=your-key in your environment or .env file
ox = OpenXtract(model="openai:gpt-4o")

# Pass API key directly
ox = OpenXtract(
    model="openai:gpt-4o",
    api_key="sk-your-api-key-here"
)

# Use model without colon when provider or base_url is specified
ox = OpenXtract(
    model="gpt-4o",
    provider="openai",
    api_key="sk-your-api-key-here"
)

# Or with custom base URL
ox = OpenXtract(
    model="gpt-4o",
    api_key="sk-your-api-key-here",
    base_url="https://api.openai.com/v1"
)

# Parameters take priority over environment variables
# This will use "direct-key" even if OPENAI_API_KEY is set
ox = OpenXtract(
    model="openai:gpt-4o",
    api_key="direct-key"
)
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

## Use Cases

- **Document Processing**: Extract data from invoices, receipts, contracts
- **Data Migration**: Convert unstructured legacy data to structured formats
- **Content Analysis**: Parse emails, reports, and documents for key information
- **Business Automation**: Automate data entry from various document types
- **Form Processing**: Extract form data from scanned documents and images

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT - see [LICENSE](LICENSE).

---

Built by [Mellow AI](https://github.com/Mellow-Artificial-Intelligence)
