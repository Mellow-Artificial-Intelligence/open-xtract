# open-xtract-v2

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![pydantic-ai](https://img.shields.io/badge/pydantic--ai-1.37+-7C3AED.svg)](https://ai.pydantic.dev/)

Extract structured data from documents, images, audio, and video using LLMs.

## Installation

```bash
uv sync
```

## Usage

```python
from pydantic import BaseModel
from main import extract

class PdfInfo(BaseModel):
    summary: str
    language: str

result = extract(
    schema=PdfInfo,
    model='google-gla:gemini-3-flash-preview',
    url='https://example.com/document.pdf',
    instructions="return a 2 sentence summary and the primary language of the document",
)
print(result)
```

## Supported Media Types

| Type | Extensions |
|------|------------|
| Documents | `.pdf`, `.doc`, `.docx`, `.txt`, `.html`, `.csv`, `.xls`, `.xlsx` |
| Images | `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.svg` |
| Audio | `.mp3`, `.wav`, `.ogg`, `.flac`, `.aac`, `.m4a` |
| Video | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.wmv` |
