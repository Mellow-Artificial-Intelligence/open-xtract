<div align="center">

# openextract

**Extract structured data from documents, images, audio, and video using LLMs.**

[![PyPI version](https://img.shields.io/pypi/v/openextract.svg?logo=pypi&logoColor=white&color=4B8BBE)](https://pypi.org/project/openextract/)
[![Python versions](https://img.shields.io/pypi/pyversions/openextract.svg?logo=python&logoColor=white)](https://pypi.org/project/openextract/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/Mellow-Artificial-Intelligence/openextract/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://img.shields.io/pypi/dm/openextract.svg?color=blue)](https://pypi.org/project/openextract/)

[Documentation](https://mellow-artificial-intelligence.github.io/openextract/) &middot; [PyPI](https://pypi.org/project/openextract/) &middot; [Changelog](CHANGELOG.md) &middot; [Issues](https://github.com/Mellow-Artificial-Intelligence/openextract/issues)

</div>

---

`openextract` turns any document, image, audio, or video file into a typed Pydantic model in a single function call. Point it at a local path or a URL, pass a schema, and get back a validated object you can use directly in your code.

## Features

- **Type-safe output.** Define your shape with Pydantic; get back a validated instance.
- **One function, many modalities.** Documents (PDF, DOCX), images, audio, and video.
- **Local files or URLs.** Pass a path or an `https://` URL &mdash; `openextract` handles fetching.
- **Bring your own model.** OpenAI, Anthropic, Google, AWS Bedrock, xAI, Cohere, Hugging Face, Groq, Cerebras, Mistral, and Ollama supported out of the box via [`pydantic-ai`](https://github.com/pydantic/pydantic-ai).
- **Explicit error handling.** Distinct exceptions for URL fetch, schema validation, and model errors.
- **100% test coverage**, enforced in CI.

## Installation

```bash
uv add openextract
```

Or with pip:

```bash
pip install openextract
```

Model calls require a provider SDK. Install the extra for the provider you use, for example `openextract[openai]`, `openextract[anthropic]`, or `openextract[all]` for every supported provider. The base package ships `pydantic-ai-slim` without provider SDKs pre-installed. If the requested provider SDK is missing, `openextract` raises `ProviderNotInstalledError` with a provider-specific `pip install 'openextract[...]'` command when the model prefix is known.

Requires Python 3.12+.

## Quick start

```python
from pydantic import BaseModel
from openextract import extract


class PdfInfo(BaseModel):
    summary: str
    language: str


result = extract(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="https://example.com/document.pdf",
    instructions="Return a two-sentence summary and the document's primary language.",
)

print(result.summary)
print(result.language)
```

`result` is a fully-validated `PdfInfo` instance &mdash; not a dict, not a string.

## Usage

### Local files

```python
result = extract(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="./reports/q4.pdf",
)
```

### Bytes or file-like objects

```python
result = extract(schema=PdfInfo, model="xai:grok-4.3", input_file=pdf_bytes, media_type="application/pdf")
# A file-like object with .read() works too; pass media_type explicitly:
result = extract(schema=PdfInfo, model="xai:grok-4.3", input_file=open("q4.pdf", "rb"), media_type="application/pdf")
```

### Input size limits

Every input is capped at 50 MiB (`52_428_800` bytes) before a model call.
Local paths are checked before and during reading, URL bodies are streamed
through the cap even when `Content-Length` is missing or incorrect, and binary
streams are read in bounded chunks. Override the limit for one call with
`max_input_bytes`, or set `OPENEXTRACT_MAX_INPUT_BYTES` for the process:

```python
result = extract(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="./reports/large.pdf",
    max_input_bytes=100 * 1024 * 1024,
)
```

Oversized inputs raise `InputTooLargeError` before a model request. The CLI
exposes the same control as `--max-input-bytes` and reports the error with exit
code `5`.

### Retry on transient model errors

```python
result = extract(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="./reports/q4.pdf",
    max_retries=3,
)
```

`max_retries` defaults to `0` (single attempt) and must be a non-negative integer.
Only transient `ModelError` failures—timeouts, rate limits, and supported 5xx
responses—are retried; authentication, permission, and invalid-request failures
fail immediately. Delays use exponential backoff with up to 25% additive jitter,
bounded by `retry_max_backoff` (60 seconds by default). A valid provider
`Retry-After` value takes precedence but is still bounded. Both backoff values
must be finite and non-negative.

The input is resolved once before the first model attempt. Retries reuse the same
media bytes, prompt, and agent, so URLs and non-seekable streams are not fetched
or read again.

### Inspecting token usage

Use `extract_with_usage` when you want token counts alongside the extracted output (for cost tracking, logging, etc.).

```python
from openextract import extract_with_usage

result, usage = extract_with_usage(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="./reports/q4.pdf",
)

print(result.summary)
print(f"tokens: {usage.input_tokens} in / {usage.output_tokens} out / {usage.total_tokens} total")
```

`usage` is a frozen `Usage` dataclass with `input_tokens`, `output_tokens`, and `total_tokens` fields.


### Choosing a model

`model` follows the `pydantic-ai` provider prefix convention:

| Provider     | Example identifier                                       | Install extra |
| ------------ | -------------------------------------------------------- | ------------- |
| OpenAI       | `openai:gpt-5`                                           | `openextract[openai]` |
| Anthropic    | `anthropic:claude-sonnet-4`                              | `openextract[anthropic]` |
| Google       | `google-gla:gemini-2.5-pro`                              | `openextract[google]` |
| AWS Bedrock  | `bedrock:anthropic.claude-sonnet-4-20250514-v1:0`        | `openextract[bedrock]` |
| xAI          | `xai:grok-4.3`                                           | `openextract[xai]` |
| Cohere       | `cohere:command-r-plus`                                  | `openextract[cohere]` |
| Hugging Face | `huggingface:meta-llama/Llama-3.3-70B-Instruct`          | `openextract[huggingface]` |
| Groq         | `groq:llama-3.3-70b-versatile`                           | `openextract[groq]` |
| Cerebras     | `cerebras:llama3.1-70b`                                  | `openextract[openai]` |
| Mistral      | `mistral:mistral-large-latest`                           | `openextract[mistral]` |
| OpenRouter   | `openrouter:anthropic/claude-sonnet-4`                   | `openextract[openrouter]` |
| Outlines     | `outlines:transformers/meta-llama/Llama-3.2-1B-Instruct` | Install the matching `pydantic-ai-slim[outlines-*]` backend |
| Ollama       | `ollama:llama3`                                          | `openextract[openai]` |

OpenAI identifiers using the concise `openai:` prefix are routed through the
Responses API by default. Use `openai-responses:` to select it explicitly or
`openai-chat:` to force the legacy Chat Completions API.

Ollama and Cerebras work via the `openai`-compatible code path &mdash; no dedicated extra is required for either.

Set the corresponding provider credentials in your environment (e.g. `XAI_API_KEY` for xAI). `openextract` loads `.env` automatically.

OpenRouter and Cerebras are openai-compatible (they go through the `openai` client under the hood), so their errors are already classified via the existing openai path &mdash; no separate exception handling is needed.

Outlines runs models locally (via HuggingFace transformers, llama-cpp, MLX, vLLM, or SGLang) and enforces JSON-schema-conforming output at the token level. Install it separately alongside the backend you want, for example `pip install pydantic-ai-slim[outlines-transformers]`.

### Command line

`openextract` ships with a CLI for one-shot extractions from the shell.

```bash
openextract ./reports/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --instructions "Pull totals and line items." \
  --output json
```

Batch multiple files (JSON array output):

```bash
openextract ./invoices/a.pdf ./invoices/b.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3
```

Token usage (single file):

```bash
openextract ./reports/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --usage
```

Read from stdin:

```bash
cat ./reports/q4.pdf | openextract - \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --media-type application/pdf
```

- `input_file` accepts one or more paths/URLs, or `-` for stdin (`--media-type` required for stdin).
- `--schema` is a Python import path of the form `module:ClassName` resolving to a Pydantic model.
- `--model` is a `pydantic-ai` model identifier.
- `--instructions` is optional natural-language guidance.
- `--media-type` sets MIME type for stdin or overrides guessing for paths/URLs.
- `--usage` prints a JSON object with `result` and `usage` (single input only).
- `--output` is `json` (default) or `repr`.
- `--max-retries`, `--retry-backoff`, and `--retry-max-backoff` match the Python
  API retry behavior.
- `--max-input-bytes` overrides the 50 MiB per-input cap.
- `--continue-on-error` (batch only) keeps processing when an input fails; each
  failure is emitted inline as `{"input", "error", "error_type"}` and the command
  exits `7` if any input failed. Without it, a batch aborts on the first failure.

Exit codes: `0` success, `2` URL fetch error, `3` schema validation error, `4` model error,
`5` other extraction error, `6` missing provider extra, `7` partial batch failure
(`--continue-on-error`), `1` any other failure (including bad `--schema` paths).

Extraction errors are written to stderr; successful JSON, usage payloads, and
`--continue-on-error` batch arrays are written to stdout. Missing provider extras
exit `6` and include the same install hint as the Python API, for example
`pip install 'openextract[xai]'`. Partial batch failures with `--continue-on-error`
still print the full batch array to stdout, write a warning to stderr, and exit `7`.

Full stdout/stderr/exit-code contracts: [docs/cli.md](docs/cli.md).
Provider capability matrix: [docs/providers.md](docs/providers.md).
Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md).

## Examples

Runnable scripts live in [`examples/`](examples/), grouped by use case (local files, bytes, URLs, images, batch, async, retries, CLI, and more). See [examples/README.md](examples/README.md) for the full table.

```bash
# Run all fixture-based examples (uses OpenAI, Anthropic, and xAI — see examples/README.md)
uv run python examples/run_all.py

# Single example with the bundled sample image
uv run python examples/basic/local_file.py --fixture
```

[See the examples/ directory](examples/) for the full source.

### Error handling

```python
from openextract import (
    extract,
    InputTooLargeError,
    UrlFetchError,
    SchemaValidationError,
    ModelError,
    ProviderNotInstalledError,
    ExtractionError,
)

try:
    result = extract(schema=PdfInfo, model="xai:grok-4.3", input_file=url)
except UrlFetchError:
    ...  # The URL could not be fetched
except InputTooLargeError:
    ...  # The input exceeded the configured byte limit
except SchemaValidationError:
    ...  # The model's output did not match your schema
except ProviderNotInstalledError:
    ...  # The provider extra isn't installed (e.g. pip install openextract[xai])
except ModelError as exc:
    # Structured fields are populated when the provider exposes them.
    print(exc.provider, exc.status_code, exc.retryable, exc.retry_after)
except ExtractionError:
    ...  # Any other extraction failure (base class)
```

All `openextract` exceptions inherit from `ExtractionError`, so you can catch it as a single fallback if you prefer.

## API reference

### `extract(schema, model, input_file, instructions=None, *, media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

| Argument        | Type                          | Description                                                                                                       |
| --------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `schema`        | `type[BaseModel]`             | A Pydantic model class describing the desired output shape.                                                       |
| `model`         | `str`                         | A `pydantic-ai` model identifier (e.g. `"xai:grok-4.3"`).                                                         |
| `input_file`    | `str \| bytes \| BinaryIO`    | A local file path, an `https://` URL, raw `bytes`, or a binary file-like object with a `.read()` method.          |
| `instructions`  | `str \| None`                 | Optional natural-language guidance for the model.                                                                 |
| `media_type`    | `str \| None` (keyword-only)  | MIME type. Required for `bytes` and file-like inputs; overrides the guessed type for `str` inputs when provided.  |
| `max_input_bytes` | `int \| None` (keyword-only) | Per-input byte cap. `None` uses the environment or 50 MiB default. |
| `max_retries`   | `int` (keyword-only)          | Extra attempts after a transient `ModelError`. Must be a non-negative integer. Defaults to `0` (no retry).       |
| `retry_backoff` | `float` (keyword-only)        | Base seconds for exponential backoff with jitter. Must be non-negative and finite.                               |
| `retry_max_backoff` | `float` (keyword-only)    | Maximum retry delay, including `Retry-After`. Must be non-negative and finite. Defaults to `60.0`.               |

Returns an instance of `schema`.

### `extract_async(schema, model, input_file, instructions=None, *, media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Async counterpart to `extract`. Uses `Agent.run` instead of `run_sync` and accepts the same arguments.

Returns an instance of `schema`.

### `extract_with_usage(schema, model, input_file, instructions=None, *, media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Like `extract`, but returns `(output, Usage)` where `Usage` is a frozen dataclass with `input_tokens`, `output_tokens`, and `total_tokens`. Useful for cost tracking and logging. Uses the same `ModelError` retry behavior as `extract`.

### `extract_with_usage_async(schema, model, input_file, instructions=None, *, media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Async sibling of `extract_with_usage`; returns `(output, Usage)`.

### `extract_many(schema, model, input_files, instructions=None, *, media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Run concurrent extractions from synchronous code. Each item in `input_files` is a path, URL, `bytes`, or file-like object (same rules as `extract`). Results are returned in input order.

Do not call `extract_many` from a running event loop (for example inside
`async def` or a notebook with an active loop). It uses `asyncio.run` and raises
`RuntimeError` with a pointer to `extract_many_async`. Async callers should use
`await extract_many_async(...)`.

| Argument             | Type                         | Description                                                                 |
| -------------------- | ---------------------------- | --------------------------------------------------------------------------- |
| `input_files`        | `Iterable[str \| bytes \| BinaryIO]` | One input per extraction.                                          |
| `media_type`         | `str \| None` (keyword-only) | Applied uniformly to every item; required if any item is `bytes`/file-like. |
| `max_input_bytes`    | `int \| None` (keyword-only) | Per-item byte cap. `None` uses the environment or 50 MiB default. |
| `max_concurrency`    | `int` (keyword-only)         | Maximum in-flight extractions. Must be a positive integer. Defaults to `5`. |
| `return_exceptions`  | `bool` (keyword-only)        | If `True`, exceptions appear in the result list instead of being raised.    |
| `max_retries`        | `int` (keyword-only)         | Per-item extra attempts after a transient `ModelError`. Must be a non-negative integer. Defaults to `0`. |
| `retry_backoff`      | `float` (keyword-only)       | Base seconds for per-item exponential backoff with jitter. Must be non-negative and finite. |
| `retry_max_backoff`  | `float` (keyword-only)       | Maximum per-item retry delay, including `Retry-After`. Defaults to `60.0`. |

Returns a `list` of schema instances (or exceptions when `return_exceptions=True`).

### `extract_many_async(schema, model, input_files, instructions=None, *, media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Async sibling of `extract_many`; same arguments and return shape.

### `Usage`

Frozen dataclass returned by `extract_with_usage` / `extract_with_usage_async`:

| Field            | Type  | Description              |
| ---------------- | ----- | ------------------------ |
| `input_tokens`   | `int` | Prompt tokens consumed.  |
| `output_tokens`  | `int` | Completion tokens.       |
| `total_tokens`   | `int` | Total tokens for the call. |

## Public API stability

`openextract.__all__` is the public Python API surface. Modules and helpers whose
names start with `_`, including `openextract._extract` and `openextract._cli`, are
internal implementation details. The CLI command is also user-facing and follows
the compatibility notes below even though it is not exported from `__all__`.

| API | Status for 1.0 | Notes |
| --- | --- | --- |
| `extract` | Stable | Primary synchronous API. Signature, return type, media input forms, retry behavior, and public exception categories are intended to carry into 1.0 unchanged. |
| `extract_async` | Stable | Async sibling of `extract`; same input contract and retry behavior, with `Agent.run` instead of `run_sync`. |
| `extract_with_usage` | Stable | Usage-returning sync API. The `(output, Usage)` tuple shape is stable; exact token values depend on provider reporting. |
| `extract_with_usage_async` | Stable | Async sibling of `extract_with_usage`; same tuple shape and retry behavior. |
| `extract_many` | Provisional | Batch return ordering, option validation, and `return_exceptions` semantics are intended to remain. Calling it from a running event loop raises `RuntimeError`; use `extract_many_async` in async code. |
| `extract_many_async` | Provisional | Async batch API with the same return shape, option constraints, and per-item retry behavior as `extract_many`. |
| `Usage` | Stable | Frozen dataclass with `input_tokens`, `output_tokens`, and `total_tokens`. New fields, if ever needed, should be additive. |
| `ExtractionError` | Stable | Base class for all public `openextract` exceptions. Catch this for a broad fallback. |
| `UrlFetchError` | Stable | Raised for URL fetch and URL safety failures. Message wording may improve, but the exception type is stable. |
| `InputTooLargeError` | Stable | Raised before a model call when resolved media exceeds the configured per-input byte limit. |
| `SchemaValidationError` | Stable | Raised when model output cannot be validated against the requested schema. |
| `ModelError` | Stable | Raised for provider/model API failures, with `provider`, `status_code`, `retryable`, and `retry_after` metadata where available. |
| `ProviderNotInstalledError` | Stable | Raised when the requested model provider extra is missing. Install hints may become more specific as providers are added. |
| `openextract` CLI | Provisional | The command, core flags, JSON output, stderr error reporting, provider-install exit code `6`, and partial-batch exit code `7` are intended to remain. |

No pre-1.0 signature changes are currently proposed for stable symbols.

## Compatibility and deprecation policy

`openextract` follows semantic-versioning intent, with extra care while the
project is still pre-1.0:

- **Public API:** `openextract.__all__` is the public Python API. The documented
  CLI arguments and exit codes, supported optional extras, documented environment
  variables, and documented input/output behavior are also user-facing
  compatibility surfaces.
- **Private API:** modules, functions, classes, and constants whose names start
  with `_` are internal unless they are explicitly documented here. They may
  change without a deprecation period.
- **Patch releases:** should fix bugs, documentation, packaging, provider error
  classification, or security issues without intentionally breaking public API.
- **Minor releases before 1.0:** may make breaking public API changes when they
  are needed for correctness, security, or a clearer long-term contract. These
  changes must be called out in `CHANGELOG.md` as breaking changes.
- **Major releases after 1.0:** are the normal place for breaking public API
  removals or incompatible behavior changes.

Deprecated public APIs should remain available until at least the next minor
release before `1.0`, unless keeping them would create a security, correctness,
or maintenance risk. After `1.0`, deprecated public APIs should remain available
until the next major release. Deprecations should be documented in
`CHANGELOG.md` with the replacement path and the earliest expected removal
version when that is known.

Provider behavior depends partly on `pydantic-ai` and provider SDKs. Upstream
model availability, credential requirements, supported media types, token usage
reporting, and provider-specific error shapes can change outside an
`openextract` release. `openextract` aims to keep its own public contract stable,
but provider-specific compatibility notes may be updated as upstream behavior
changes.

Python support follows `requires-python` in `pyproject.toml`; the current minimum
is Python 3.12. Dropping support for a Python minor version is a breaking change
and should be announced in `CHANGELOG.md`.

## Security

### URL fetching and SSRF

When `input_file` is an `http://` or `https://` URL, `openextract` fetches it
with host validation, redirect re-checks, and configurable timeout/redirect
limits. Summary:

- Supported schemes: `http://`, `https://`
- Non-public hosts (private, loopback, link-local/metadata, multicast, reserved)
  are refused unless `OPENEXTRACT_ALLOW_PRIVATE_URLS` is set
- Hosts are re-validated at every redirect hop
- `OPENEXTRACT_URL_TIMEOUT` (default `30`) and `OPENEXTRACT_MAX_REDIRECTS`
  (default `10`) tune fetch behavior
- `OPENEXTRACT_MAX_INPUT_BYTES` (default `52428800`) caps each resolved input,
  including streamed URL bodies with missing or incorrect length headers

Full model, remaining risk boundaries (including DNS rebinding), and reporting
process: [SECURITY.md](SECURITY.md#url-input-security-model).

## Development

```bash
git clone https://github.com/Mellow-Artificial-Intelligence/openextract.git
cd openextract
uv sync --dev

uv run pytest --cov=openextract            # tests + coverage
uv run ruff check .                        # lint (Astral ruff)
uv run ruff format --check .               # format check
uv run ty check                            # types (Astral ty)
```

CI runs the test suite on every PR and fails if total coverage drops below 100%.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

## Roadmap

The project roadmap lives in the [GitHub Wiki](https://github.com/Mellow-Artificial-Intelligence/openextract/wiki/Roadmap).

## License

[MIT](LICENSE) &copy; Cole McIntosh
