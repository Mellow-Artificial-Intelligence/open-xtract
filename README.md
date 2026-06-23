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

Model calls require a provider SDK. Install the extra for the provider you use, for example `openextract[openai]`, `openextract[anthropic]`, or `openextract[all]` for every supported provider. The base package ships `pydantic-ai-slim` without provider SDKs pre-installed.

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

### Retry on transient model errors

```python
result = extract(
    schema=PdfInfo,
    model="xai:grok-4.3",
    input_file="./reports/q4.pdf",
    max_retries=3,
)
```

`max_retries` defaults to `0` (single attempt) and must be a non-negative integer. When set, `extract` retries only on `ModelError` and sleeps `retry_backoff * (2 ** attempt)` seconds (with up to 25% jitter) between attempts. `retry_backoff` defaults to `1.0` second and must be positive and finite.

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

| Provider     | Example identifier                                       |
| ------------ | -------------------------------------------------------- |
| OpenAI       | `openai:gpt-5`                                           |
| Anthropic    | `anthropic:claude-sonnet-4`                              |
| Google       | `google-gla:gemini-2.5-pro`                              |
| AWS Bedrock  | `bedrock:anthropic.claude-sonnet-4-20250514-v1:0`        |
| xAI          | `xai:grok-4.3`                                             |
| Cohere       | `cohere:command-r-plus`                                  |
| Hugging Face | `huggingface:meta-llama/Llama-3.3-70B-Instruct`          |
| Groq         | `groq:llama-3.3-70b-versatile`                           |
| Cerebras     | `cerebras:llama3.1-70b`                                  |
| Mistral      | `mistral:mistral-large-latest`                           |
| OpenRouter   | `openrouter:anthropic/claude-sonnet-4`                   |
| Outlines     | `outlines:transformers/meta-llama/Llama-3.2-1B-Instruct` |
| Ollama       | `ollama:llama3`                                          |

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
- `--max-retries` / `--retry-backoff` match the Python API retry behavior.
- `--continue-on-error` (batch only) keeps processing when an input fails; each
  failure is emitted inline as `{"input", "error", "error_type"}` and the command
  exits `7` if any input failed. Without it, a batch aborts on the first failure.

Exit codes: `0` success, `2` URL fetch error, `3` schema validation error, `4` model error,
`5` other extraction error, `6` missing provider extra, `7` partial batch failure
(`--continue-on-error`), `1` any other failure (including bad `--schema` paths).

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
except SchemaValidationError:
    ...  # The model's output did not match your schema
except ProviderNotInstalledError:
    ...  # The provider extra isn't installed (e.g. pip install openextract[xai])
except ModelError:
    ...  # The model provider returned an error
except ExtractionError:
    ...  # Any other extraction failure (base class)
```

All `openextract` exceptions inherit from `ExtractionError`, so you can catch it as a single fallback if you prefer.

## API reference

### `extract(schema, model, input_file, instructions=None, *, media_type=None, max_retries=0, retry_backoff=1.0)`

| Argument        | Type                          | Description                                                                                                       |
| --------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `schema`        | `type[BaseModel]`             | A Pydantic model class describing the desired output shape.                                                       |
| `model`         | `str`                         | A `pydantic-ai` model identifier (e.g. `"xai:grok-4.3"`).                                                         |
| `input_file`    | `str \| bytes \| BinaryIO`    | A local file path, an `https://` URL, raw `bytes`, or a binary file-like object with a `.read()` method.          |
| `instructions`  | `str \| None`                 | Optional natural-language guidance for the model.                                                                 |
| `media_type`    | `str \| None` (keyword-only)  | MIME type. Required for `bytes` and file-like inputs; overrides the guessed type for `str` inputs when provided.  |
| `max_retries`   | `int` (keyword-only)          | Extra attempts after a `ModelError`. Must be a non-negative integer. Defaults to `0` (no retry).                 |
| `retry_backoff` | `float` (keyword-only)        | Base seconds for exponential backoff with jitter between retries. Must be positive and finite.                   |

Returns an instance of `schema`.

### `extract_async(schema, model, input_file, instructions=None, *, media_type=None, max_retries=0, retry_backoff=1.0)`

Async counterpart to `extract`. Uses `Agent.run` instead of `run_sync`. Accepts the same `schema`, `model`, `input_file`, `instructions`, `media_type`, `max_retries`, and `retry_backoff` arguments.

Returns an instance of `schema`.

### `extract_with_usage(schema, model, input_file, instructions=None, *, media_type=None, max_retries=0, retry_backoff=1.0)`

Like `extract`, but returns `(output, Usage)` where `Usage` is a frozen dataclass with `input_tokens`, `output_tokens`, and `total_tokens`. Useful for cost tracking and logging. Uses the same `ModelError` retry behavior as `extract`.

### `extract_with_usage_async(schema, model, input_file, instructions=None, *, media_type=None, max_retries=0, retry_backoff=1.0)`

Async sibling of `extract_with_usage`; returns `(output, Usage)`.

### `extract_many(schema, model, input_files, instructions=None, *, media_type=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0)`

Run concurrent extractions from synchronous code. Each item in `input_files` is a path, URL, `bytes`, or file-like object (same rules as `extract`). Results are returned in input order.

| Argument             | Type                         | Description                                                                 |
| -------------------- | ---------------------------- | --------------------------------------------------------------------------- |
| `input_files`        | `Iterable[str \| bytes \| BinaryIO]` | One input per extraction.                                          |
| `media_type`         | `str \| None` (keyword-only) | Applied uniformly to every item; required if any item is `bytes`/file-like. |
| `max_concurrency`    | `int` (keyword-only)         | Maximum in-flight extractions. Must be a positive integer. Defaults to `5`. |
| `return_exceptions`  | `bool` (keyword-only)        | If `True`, exceptions appear in the result list instead of being raised.    |
| `max_retries`        | `int` (keyword-only)         | Per-item extra attempts after a `ModelError`. Must be a non-negative integer. Defaults to `0`. |
| `retry_backoff`      | `float` (keyword-only)       | Base seconds for per-item exponential backoff with jitter. Must be positive and finite. |

Returns a `list` of schema instances (or exceptions when `return_exceptions=True`).

### `extract_many_async(schema, model, input_files, instructions=None, *, media_type=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0)`

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
| `extract_many` | Provisional | Batch return ordering, option validation, and `return_exceptions` semantics are intended to remain, but pre-1.0 work should clarify behavior when called inside an active event loop. |
| `extract_many_async` | Provisional | Async batch API with the same return shape, option constraints, and per-item retry behavior as `extract_many`. |
| `Usage` | Stable | Frozen dataclass with `input_tokens`, `output_tokens`, and `total_tokens`. New fields, if ever needed, should be additive. |
| `ExtractionError` | Stable | Base class for all public `openextract` exceptions. Catch this for a broad fallback. |
| `UrlFetchError` | Stable | Raised for URL fetch and URL safety failures. Message wording may improve, but the exception type is stable. |
| `SchemaValidationError` | Stable | Raised when model output cannot be validated against the requested schema. |
| `ModelError` | Stable | Raised for provider/model API failures. Provider-specific classifiers may expand without changing the public type. |
| `ProviderNotInstalledError` | Stable | Raised when the requested model provider extra is missing. Install hints may become more specific as providers are added. |
| `openextract` CLI | Provisional | The command, core flags, JSON output, and documented exit-code categories are intended to remain, but provider-install and partial-batch failure behavior should finish release validation before 0.9. |

No pre-1.0 signature changes are currently proposed for stable symbols. Known
pre-1.0 follow-ups are limited to active-event-loop behavior and final CLI
release readiness.

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
directly. To reduce server-side request forgery risk when callers pass
untrusted URLs, the fetcher refuses any URL whose host resolves to a
non-public address &mdash; private RFC 1918 ranges, loopback, link-local (including
the `169.254.169.254` cloud-metadata endpoint), multicast, and reserved
ranges, for both IPv4 and IPv6 (including IPv4-mapped IPv6 like
`::ffff:127.0.0.1`). The host is re-validated at every redirect hop, so an
attacker cannot use a public URL that redirects to an internal one.

For workflows that legitimately need to fetch internal URLs (testing
against `localhost`, on-prem services, etc.), set the
`OPENEXTRACT_ALLOW_PRIVATE_URLS` environment variable to `1`, `true`, or
`yes` to disable the check.

Tune fetch behavior with:

- `OPENEXTRACT_URL_TIMEOUT` &mdash; HTTP timeout in seconds (default `30`)
- `OPENEXTRACT_MAX_REDIRECTS` &mdash; maximum redirect hops (default `10`)

Invalid or non-positive values fall back to the defaults. If you need a one-off fetch from an internal
host without disabling validation globally, fetch the bytes with your own
HTTP client and pass them to `extract()` as `bytes`/file-like with an
explicit `media_type`.

> **Note:** host validation is best-effort; it does not defend against DNS
> rebinding (where the host resolves to different IPs across calls). Treat
> URL-based extraction of untrusted input as a privileged operation.

### Reporting vulnerabilities

See [SECURITY.md](SECURITY.md).

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
