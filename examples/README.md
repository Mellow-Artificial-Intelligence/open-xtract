# openextract examples

Runnable scripts showing common ways to use openextract. Each example prints JSON from a validated Pydantic model.

## Prerequisites

```bash
uv sync --dev
```

Set credentials for your provider (openextract loads `.env` automatically). Examples pick a model automatically:

1. `OPENEXTRACT_MODEL` if set (e.g. `openrouter:openai/gpt-4o-mini`)
2. Otherwise the first detected API key (`XAI_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, …)

Install the matching extra when needed, e.g. `uv add 'openextract[openai]'`.

## Run everything

```bash
uv run python examples/run_all.py
```

`advanced/error_handling.py` runs without API keys. The rest call your configured model against bundled fixtures in `fixtures/`.

## Examples by use case

| Directory | Script | What it demonstrates |
| --------- | ------ | -------------------- |
| `basic/` | `local_file.py` | `extract()` with a local path (`--fixture` uses the sample image) |
| `basic/` | `bytes_input.py` | `extract()` with `bytes` + `media_type` |
| `basic/` | `url_extract.py` | `extract()` with a public HTTPS URL |
| `images/` | `document_summary.py` | Summarize a document page image |
| `images/` | `receipt_extraction.py` | Receipt-style fields from an image |
| `documents/` | `invoice_extraction.py` | Invoice schema from PDF or image (`--fixture`) |
| `batch/` | `batch_extract.py` | Concurrent `extract_many()` |
| `async/` | `async_extract.py` | `extract_async()` |
| `advanced/` | `extract_with_usage.py` | `extract_with_usage()` and token counts |
| `advanced/` | `retry_extract.py` | `max_retries` / `retry_backoff` |
| `advanced/` | `error_handling.py` | Catching `UrlFetchError` (no model call) |
| `audio/` | `meeting_notes.py` | Audio → structured meeting notes (bring your own file) |
| `cli/` | `schemas.py` | Pydantic models for CLI `--schema` |

### Quick commands

```bash
# Local file (default: pass your own path)
uv run python examples/basic/local_file.py --fixture

# Bytes + explicit media type (uses bundled fixture internally)
uv run python examples/basic/bytes_input.py

# Public URL
uv run python examples/basic/url_extract.py

# Token usage
uv run python examples/advanced/extract_with_usage.py --fixture

# Batch
uv run python examples/batch/batch_extract.py

# CLI (from repo root; PYTHONPATH so examples.cli.schemas resolves)
PYTHONPATH=. uv run openextract examples/fixtures/document_page.png \
  --schema examples.cli.schemas:DocumentInfo \
  --model "${OPENEXTRACT_MODEL:-openrouter:openai/gpt-4o-mini}" \
  --instructions "Two-sentence summary and primary language."
```

### Audio

```bash
uv run python examples/audio/meeting_notes.py /path/to/meeting.mp3
```

### Your own PDF invoice

```bash
uv run python examples/documents/invoice_extraction.py ./invoices/acme-q4.pdf
```

Some providers require minimum balance for PDF uploads; image fixtures work with vision-capable chat models such as `openrouter:openai/gpt-4o-mini`.