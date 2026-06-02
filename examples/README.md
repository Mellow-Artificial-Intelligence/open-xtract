# openextract examples

Runnable scripts showing common ways to use openextract. Each example prints JSON from a validated Pydantic model.

Examples use **OpenAI**, **Anthropic**, and **xAI** so you can see how provider identifiers map to real calls. Set the matching API keys in `.env` (openextract loads them automatically).

| Provider   | Model identifier              | Environment variable   |
| ---------- | ----------------------------- | ---------------------- |
| OpenAI     | `openai:gpt-4o-mini`          | `OPENAI_API_KEY`       |
| Anthropic  | `anthropic:claude-sonnet-4`   | `ANTHROPIC_API_KEY`    |
| xAI        | `xai:grok-4.3`                | `XAI_API_KEY`          |

Install provider extras as needed: `openextract[openai]`, `openextract[anthropic]`, `openextract[xai]`, or `openextract[all]`.

Set `OPENEXTRACT_MODEL` to override every example with a single model (useful for CI or one-off testing).

## Prerequisites

```bash
uv sync --dev
```

## Run everything

```bash
uv run python examples/run_all.py
```

`advanced/error_handling.py` runs without API keys. The rest need the API key for their assigned provider (see table below), or set `OPENEXTRACT_MODEL` to run all with one model.

## Examples by use case

| Directory | Script | Provider | What it demonstrates |
| --------- | ------ | -------- | -------------------- |
| `basic/` | `local_file.py` | OpenAI | `extract()` with a local path (`--fixture` uses the sample image) |
| `basic/` | `bytes_input.py` | Anthropic | `extract()` with `bytes` + `media_type` |
| `basic/` | `url_extract.py` | xAI | `extract()` with a public HTTPS URL |
| `images/` | `document_summary.py` | xAI | Summarize a document page image |
| `images/` | `receipt_extraction.py` | Anthropic | Receipt-style fields from an image |
| `documents/` | `invoice_extraction.py` | Anthropic | Invoice schema from PDF or image (`--fixture`) |
| `batch/` | `batch_extract.py` | OpenAI | Concurrent `extract_many()` |
| `async/` | `async_extract.py` | Anthropic | `extract_async()` |
| `advanced/` | `extract_with_usage.py` | xAI | `extract_with_usage()` and token counts |
| `advanced/` | `retry_extract.py` | OpenAI | `max_retries` / `retry_backoff` |
| `advanced/` | `error_handling.py` | — | Catching `UrlFetchError` (no model call) |
| `audio/` | `meeting_notes.py` | xAI | Audio → structured meeting notes (bring your own file) |
| `cli/` | `schemas.py` | — | Pydantic models for CLI `--schema` |

### Quick commands

```bash
# OpenAI — local file
uv run python examples/basic/local_file.py --fixture

# Anthropic — bytes + explicit media type
uv run python examples/basic/bytes_input.py

# xAI — public URL
uv run python examples/basic/url_extract.py

# xAI — token usage
uv run python examples/advanced/extract_with_usage.py --fixture

# OpenAI — batch
uv run python examples/batch/batch_extract.py

# Anthropic via CLI (from repo root)
PYTHONPATH=. uv run openextract examples/fixtures/document_page.png \
  --schema examples.cli.schemas:DocumentInfo \
  --model anthropic:claude-sonnet-4 \
  --instructions "Two-sentence summary and primary language."
```

### Audio

```bash
uv run python examples/audio/meeting_notes.py /path/to/meeting.mp3
```

Requires `XAI_API_KEY` (or override with `OPENEXTRACT_MODEL`).

### Your own PDF invoice

```bash
uv run python examples/documents/invoice_extraction.py ./invoices/acme-q4.pdf
```

Uses Anthropic by default. Vision-capable models work with the image fixture; PDFs may need provider-specific balance or limits.