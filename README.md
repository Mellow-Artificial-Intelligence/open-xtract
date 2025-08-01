# OPEN-XTRACT

<div align="center">

**FROM RAW PDFS TO CLEAN, QUERYABLE TEXT**

*Open-source framework that extracts structured data from PDFs. Bring your own models and extend to any file type.*

[![GitHub stars](https://img.shields.io/github/stars/colesmcintosh/open-xtract?style=social)](https://github.com/colesmcintosh/open-xtract)
[![PyPI version](https://badge.fury.io/py/open-xtract.svg)](https://badge.fury.io/py/open-xtract)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

[**Star on GitHub**](https://github.com/colesmcintosh/open-xtract) • [**Install from PyPI**](https://pypi.org/project/open-xtract/)

</div>

---

## Quick Start

```bash
pip install open-xtract

# With vision support for multimodal models
pip install "open-xtract[vision]"
```

### Basic Usage

```python
from open_xtract import PDFProcessor
from pydantic import BaseModel

# Define your schema
class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    vendor: str

# Initialize processor
processor = PDFProcessor(
    llm_provider="openai",
    model="gpt-4o-mini"
)

# Extract data
result = processor.process_pdf(
    pdf_path="invoice.pdf",
    schema=InvoiceData
)

if result.success:
    print(f"Invoice #: {result.data.invoice_number}")
    print(f"Total: ${result.data.total_amount}")
```

### CLI Usage

```bash
# Extract with JSON schema
open-xtract invoice.pdf --schema invoice_schema.json

# Extract with inline schema
open-xtract document.pdf --schema '{"properties": {"title": {"type": "string"}}}'

# Use custom provider
open-xtract paper.pdf --schema schema.json --provider azure --base-url https://your-resource.openai.azure.com
```

## Capabilities

### **MODEL AGNOSTIC**
Bring your own OCR or LLM. open-xtract abstracts the extraction layer so you can swap engines at will.

- **WORKS WITH ANY MODEL** - Seamlessly integrate your preferred AI models
- **SIMPLE ADAPTER API** - Clean, intuitive interface for model integration  
- **OPEN SOURCE MIT** - Complete freedom to modify and distribute

### **PDF-FIRST INGESTION**
Drop in a PDF and receive clean, layout-aware text that's ready for embeddings.

- **LAYOUT-AWARE PARSING** - Preserves document structure and formatting using LangChain's PyPDFLoader
- **VISION-ENHANCED EXTRACTION** - Automatically uses multimodal models for better accuracy with complex layouts
- **HANDLES SCANNED PDFS** - Extract text from images and scanned documents with vision models

### **CITED RETRIEVAL**
Every chunk is embedded into a vector DB, reranked, and served via RAG with inline citations.

- **VECTOR SEARCH** - Semantic similarity matching
- **RERANKED ANSWERS** - Improved relevance scoring
- **INLINE CITATIONS** - Transparent source attribution

## Why Open-Xtract?

<div align="center">

| **MIT OPEN SOURCE** | **PDF FIRST FORMAT** | **FAST EXTRACTION** | **OPEN EXTENSIBLE** |
|:---:|:---:|:---:|:---:|
| Complete freedom to modify and distribute | Purpose-built for document processing | Optimized for speed and efficiency | Plugin architecture for any file type |

</div>

## Installation & Usage

### Install from PyPI
```bash
pip install open-xtract
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap

- [ ] Core PDF extraction framework
- [ ] Model-agnostic adapter API
- [ ] Vector search with citations
- [ ] Image extraction support
- [ ] Video content processing
- [ ] Advanced OCR integrations
- [ ] Cloud deployment templates

## Support & Community

- **Website**: [open-xtract.com](https://www.open-xtract.com/)
- **Documentation**: Coming soon
- **Issues**: [GitHub Issues](https://github.com/colesmcintosh/open-xtract/issues)
- **Discussions**: [GitHub Discussions](https://github.com/colesmcintosh/open-xtract/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- [Homepage](https://www.open-xtract.com/)
- [PyPI Package](https://pypi.org/project/open-xtract/)
- [Privacy Policy](https://www.open-xtract.com/privacy)
- [Terms of Service](https://www.open-xtract.com/terms)
- [Security](https://www.open-xtract.com/security)

---

<div align="center">

**© 2025 OPEN-XTRACT. ALL RIGHTS RESERVED.**

*Made with love for the open source community*

</div>
