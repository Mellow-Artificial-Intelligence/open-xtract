---
layout: page
title: API reference
---

# API reference

This is the canonical reference for the public Python API. CI compares every
function heading below with the installed callable signature; update this page
in the same change as any public signature.

## Extraction

### `extract(schema, model, input_file, instructions=None, *, media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Extract one input synchronously and return an instance of `schema`.

### `extract_async(schema, model, input_file, instructions=None, *, media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Async counterpart to `extract`. It uses `Agent.run` and returns an instance of
`schema`.

### `extract_with_usage(schema, model, input_file, instructions=None, *, media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Extract one input synchronously and return `(output, Usage)`. It has the same
retry behavior as `extract`; `Usage` describes the successful model call.

### `extract_with_usage_async(schema, model, input_file, instructions=None, *, media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Async counterpart to `extract_with_usage`; returns `(output, Usage)`.

### `extract_many(schema, model, input_files, instructions=None, *, media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Run concurrent extractions from synchronous code. Results preserve input order.
When `return_exceptions=True`, per-item exceptions appear in the result list.
Do not call this function from a running event loop; use
`extract_many_async` instead.

### `extract_many_async(schema, model, input_files, instructions=None, *, media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Async counterpart to `extract_many`; it has the same arguments, result ordering,
and per-item retry behavior.

## Common arguments

| Argument | Type | Description |
| --- | --- | --- |
| `schema` | `type[BaseModel]` | Pydantic model class describing the desired output. |
| `model` | `str` | `pydantic-ai` model identifier such as `"xai:grok-4.3"`. |
| `input_file` | `str \| bytes \| BinaryIO` | Local path, HTTP(S) URL, bytes, or binary file-like object. |
| `instructions` | `str \| None` | Optional model guidance. |
| `media_type` | `str \| None` | Required for bytes and file-like inputs; overrides inference for paths and URLs. |
| `max_input_bytes` | `int \| None` | Per-input byte cap; `None` uses `OPENEXTRACT_MAX_INPUT_BYTES` or the 50 MiB default. |
| `max_retries` | `int` | Extra attempts after transient `ModelError`; defaults to `0`. |
| `retry_backoff` | `float` | Base seconds for exponential backoff with up to 25% jitter. |
| `retry_max_backoff` | `float` | Maximum delay, including provider `Retry-After`; defaults to `60.0`. |

Batch functions also accept:

| Argument | Type | Description |
| --- | --- | --- |
| `input_files` | `Iterable[str \| bytes \| BinaryIO]` | One input per extraction. |
| `max_concurrency` | `int` | Positive maximum number of in-flight extractions; defaults to `5`. |
| `return_exceptions` | `bool` | Return per-item exceptions in place instead of failing fast. |

## `Usage`

`Usage` is a frozen dataclass returned by the two usage helpers.

| Field | Type | Description |
| --- | --- | --- |
| `input_tokens` | `int` | Prompt tokens consumed. |
| `output_tokens` | `int` | Completion tokens consumed. |
| `total_tokens` | `int` | Total reported tokens. |

## Configuration

The Python library reads provider configuration from the existing process
environment and does not load `.env` files. The `openextract` CLI and bundled
examples load `.env` explicitly as an application-level convenience.
