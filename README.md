# OPEN-XTRACT

<div align="center">

**Turn documents into structured data**

*Open‑source toolkit for extracting clean, structured data from PDFs, images, and text. Minimal setup.*

[![GitHub stars](https://img.shields.io/github/stars/colesmcintosh/open-xtract?style=social)](https://github.com/colesmcintosh/open-xtract)
[![PyPI version](https://badge.fury.io/py/open-xtract.svg)](https://badge.fury.io/py/open-xtract)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

[**Star on GitHub**](https://github.com/colesmcintosh/open-xtract) • [**Install from PyPI**](https://pypi.org/project/open-xtract/)

</div>

---

## Quick Start

```bash
uv add open-xtract

# With vision support for multimodal models
uv add 'open-xtract[vision]'
```

Set your model credentials (example for OpenAI):

```bash
export OPENAI_API_KEY=your_key
```

### Basic Programmatic Usage

```python
from pydantic import BaseModel
from open_xtract import OpenXtract

class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    vendor: str

ox = OpenXtract(model="gpt-5")

# PDF
result = ox.extract_pdf("invoice.pdf", InvoiceData)
print(result)

# Image (file path or URL)
result = ox.extract_image("https://example.com/receipt.png", InvoiceData)
print(result)

# Text
result = ox.extract_text("Total: $123.45 on 2025-03-01 from ACME", InvoiceData)
print(result)
```

### Retrieval with Citations (Optional Reranking)

Use built-in retrieval to embed your texts, answer a question, and return inline citations. This mirrors the LangChain QA citation pattern and supports FlashRank reranking.

References:
- QA with citations: [LangChain how-to](https://python.langchain.com/docs/how_to/qa_citations)
- FlashRank reranker: [LangChain integration](https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/)

```python
from pydantic import BaseModel
from open_xtract import OpenXtract

class SpeedAnswer(BaseModel):
    speed_kmh: float
    text: str

ox = OpenXtract(model="gpt-5")

texts = [
    "The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph).",
    "Average adult cheetahs weigh between 21 and 72 kg (46 and 159 lb).",
]

res = ox.retrieve(
    question="How fast are cheetahs?",
    texts=texts,
    schema=SpeedAnswer,   # REQUIRED schema for structured output
    k=4,
    rerank=True,          # set to False to disable FlashRank reranking
)

print(res.data)  # <-- conforms to SpeedAnswer
```

### Model configuration

```python
OpenXtract(
    model="gpt-5",
    base_url="https://api.openai.com/v1",
    api_key="...",
)
```

## Capabilities

- **Model agnostic**: Bring your own OCR/LLM; clean adapter surface
- **PDF-first ingestion**: Layout-aware via LangChain `PyPDFLoader`
- **Vision-enhanced**: Multimodal image parsing when needed
- **Cited retrieval**: Embed → retrieve → (optional) rerank → answer with citations

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT — see [LICENSE](LICENSE).

## Links

- [Homepage](https://www.open-xtract.com/)
- [PyPI Package](https://pypi.org/project/open-xtract/)
- [Issues](https://github.com/colesmcintosh/open-xtract/issues)

---

<div align="center">

**© 2025 OPEN-XTRACT. ALL RIGHTS RESERVED.**

*Made with love for the open source community*

</div>
