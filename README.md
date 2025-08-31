# OPEN-XTRACT

Turn documents and images into structured JSON. Minimal setup. Fast.

Built on OpenRouter structured outputs and multimodal support.

- Structured outputs: [docs](https://openrouter.ai/docs/features/structured-outputs)
- Multimodal images: [docs](https://openrouter.ai/docs/features/multimodal/images)
- Message transforms (middle-out): [docs](https://openrouter.ai/docs/features/message-transforms)

---

## Quick Start

Prereqs: Python 3.12+, [uv](https://docs.astral.sh/uv/) installed.

1) Clone and setup

```bash
git clone https://github.com/Mellow-Artificial-Intelligence/open-xtract
cd open-xtract
make setup
```

2) Configure API key

We use OpenRouter; put your key in `.env` as `OPENROUTER_API_KEY=...` (or export it).

```bash
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

3) Doctor check

```bash
make doctor
```

4) Run the API server

```bash
make run    # or: make run-dev for autoreload
```

Server runs at `http://0.0.0.0:8000`.

---

## API

POST `/extract` — extract structured data from a document or image. Auto-routes based on file type.

Request body:

```json
{
  "content_url": "https://bitcoin.org/bitcoin.pdf",
  "model": "google/gemini-2.5-flash-lite",
  "schema": {
    "name": "document_summary",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "title": {"type": "string"},
        "main_points": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["title", "main_points"],
      "additionalProperties": false
    }
  },
  "prompt": "Extract the title and main points"
}
```

Notes
- `schema` is required and follows OpenRouter’s JSON Schema format (`name`, `strict`, `schema`).
- We default message transforms to `["middle-out"]` unless you pass `"transforms": []`.
- For image URLs that a provider cannot fetch, we auto-retry by inlining as base64.

Example curl

```bash
curl -X POST http://0.0.0.0:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "content_url": "https://bitcoin.org/bitcoin.pdf",
    "model": "google/gemini-2.5-flash-lite",
    "schema": {
      "name": "document_summary",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
        "additionalProperties": false
      }
    },
    "prompt": "Extract the title"
  }'
```

Try with an image too by changing `content_url` to any public image URL.

---

## Programmatic Usage

```python
from open_xtract import Extract

document_schema = {
    "name": "document_summary",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "main_points": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title", "main_points"],
        "additionalProperties": False
    }
}

extractor = Extract(model="google/gemini-2.5-flash-lite")
result = extractor.extract(
    content_url="https://bitcoin.org/bitcoin.pdf",
    schema=document_schema,
    prompt="Extract the title and main points"
)
print(result)
```

What happens
- Auto-detects images vs documents and builds the right content payload
- Adds OpenRouter `response_format` with your schema (strict mode)
- Defaults `transforms` to `middle-out` for safer long inputs
- Retries images with base64 if provider cannot fetch the URL

---

## Dev Commands

Everything runs through `uv` via the Makefile:

- Setup env: `make setup`
- Doctor checks: `make doctor`
- Run server: `make run` (or `make run-dev`)
- Tests: `make test`
- Hit demo requests locally: `make test-api`

All commands honor `.env` (loaded in-app). Ensure `OPENROUTER_API_KEY` is set.

---

## License

MIT — see [LICENSE](LICENSE).

---

If this project helps you, please ⭐ the repo.
