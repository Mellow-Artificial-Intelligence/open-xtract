# Open Xtract

Open-source framework that extracts structured data from PDFs. Bring your own OCR or LLM and extend to any file type.

## Features
- **Model-agnostic** – simple adapter API works with any OCR engine or large language model.
- **PDF-first ingestion** – layout-aware parsing produces clean, tokenized text.
- **Cited retrieval** – vector search with reranked answers and inline citations.

## Installation
```bash
pip install open-xtract
```

## Quick Start
```python
from open_xtract import main

main()  # prints a greeting for now
```

## CLI
```bash
open-xtract
```

## License
MIT
