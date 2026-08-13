---
layout: page
title: API reference
---

# API reference

This is the canonical reference for the public Python API. CI compares every
function heading below with the installed callable signature; update this page
in the same change as any public signature.

## Extraction

### `Extractor(schema, model=None, instructions=None, *, style='direct', agent=None, model_settings=None, timeout=None, instrument=False, retry_policy=None, max_input_bytes=None, url_timeout=None)`

Reusable synchronous extraction session. Enter it with `with`; then call
`extract(input_file, *, media_type=None)` or
`extract_with_usage(input_file, *, media_type=None)`. The agent, model provider
client, and URL-fetch client are constructed once and closed on context exit.

`model` accepts either a known model string or a configured
`pydantic_ai.models.Model`. `style` selects how the model inspects the input
(`direct`, `search`, or `code`; see [Common arguments](#common-arguments)) and
applies to every call in the session; `search` and `code` sessions build their
harness agent and temporary workspace once on enter and remove both on exit.
As an advanced alternative, `agent` accepts a fully
configured Pydantic AI `Agent`; it is mutually exclusive with `model`, and its
output is revalidated against `schema`. Non-`direct` styles cannot be combined
with an injected `agent`.

### `AsyncExtractor(schema, model=None, instructions=None, *, style='direct', agent=None, model_settings=None, timeout=None, instrument=False, retry_policy=None, max_input_bytes=None, url_timeout=None)`

Async session counterpart. Enter it with `async with`; then await `extract` or
`extract_with_usage`. It shares one async HTTP client and one agent across
calls, including concurrent calls made on the entering event loop. Close it
manually with `aclose()` only when a context manager is impractical.

### `RetryPolicy(max_retries=0, backoff=1.0, max_backoff=60.0)`

Frozen session retry configuration. Only transient `ModelError` failures are
retried. The backoff and bounded provider `Retry-After` behavior match the
one-shot function arguments.

### `extract(schema, model, input_file, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Extract one input synchronously and return an instance of `schema`.

### `extract_async(schema, model, input_file, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Async counterpart to `extract`. It uses `Agent.run` and returns an instance of
`schema`.

### `extract_with_usage(schema, model, input_file, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Extract one input synchronously and return `(output, Usage)`. It has the same
retry behavior as `extract`; `Usage` describes the successful model call.

### `extract_with_usage_async(schema, model, input_file, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Async counterpart to `extract_with_usage`; returns `(output, Usage)`.

### `extract_many(schema, model, input_files, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Run concurrent extractions from synchronous code. Results preserve input order.
When `return_exceptions=True`, per-item exceptions appear in the result list.
Do not call this function from a running event loop; use
`extract_many_async` instead.

### `extract_many_async(schema, model, input_files, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Async counterpart to `extract_many`; it has the same arguments, result ordering,
and per-item retry behavior.

### `iter_extract_many_async(schema, model, input_files, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Return an async iterator of `(input_index, result)` pairs in completion order.
Inputs are consumed lazily, at most `max_concurrency` items are scheduled, and
results are available before the complete batch finishes. Simultaneous
completions are yielded in input-index order.

With `return_exceptions=False`, the first failure cancels and awaits outstanding
work before being raised. With `return_exceptions=True`, item exceptions are
yielded in the result position and streaming continues.

### `extract_many_with_results(schema, model, input_files, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Run a batch and return per-item [`ExtractionResult`](#extractionresult) objects
instead of bare schema instances. It has the same arguments, input ordering,
concurrency, and retry semantics as `extract_many`, and each result carries
token usage, attempt count, duration, model/media metadata, and a sanitized
source label. With `return_exceptions=True`, failed items appear as
`Exception` values in place. Use [`total_usage`](#total_usageresults) to
aggregate token usage across the returned results.

### `extract_many_with_results_async(schema, model, input_files, instructions=None, *, style='direct', media_type=None, max_input_bytes=None, max_concurrency=5, return_exceptions=False, max_retries=0, retry_backoff=1.0, retry_max_backoff=60.0)`

Async counterpart to `extract_many_with_results`; it has the same arguments,
result ordering, and per-item retry behavior.

### `total_usage(results)`

Sum token usage across batch extraction results, for example the list returned
by `extract_many_with_results` or `extract_many_with_results_async`. Returns a
single [`Usage`](#usage) whose fields are the totals of the successful items.

## Input and result contracts

### `ExtractionInput`

A frozen dataclass wrapping a single media source with optional per-item media
metadata. Passing one to any extraction API (or mixing them into a batch) is
equivalent to passing the raw source directly.

| Field | Type | Description |
| --- | --- | --- |
| `source` | `str \| os.PathLike[str] \| bytes \| BinaryIO` | Local path, HTTP(S) URL, `Path`, raw `bytes`, or binary file-like object. |
| `media_type` | `str \| None` | Per-item MIME type. Required for `bytes` and file-like sources when no batch-wide override is supplied. |
| `name` | `str \| None` | Optional safe source label recorded on `ExtractionResult.source`. |

Batch item media types resolve per item: an `ExtractionInput.media_type` wins
over the batch-wide `media_type` argument, which remains the fallback for raw
items.

### `ExtractionResult`

A frozen, generic dataclass returned by `extract_many_with_results*`. It never
retains raw media, credentials, query strings, fragments, or provider
internals; `source` is sanitized.

| Field | Type | Description |
| --- | --- | --- |
| `output` | `T` | The validated schema instance. |
| `usage` | [`Usage`](#usage) | Token usage from the successful model call. |
| `attempts` | `int` | Model-call attempts including retries; always `>= 1` on success. |
| `duration` | `float` | Wall-clock seconds for the item, including retries. |
| `model` | `str \| None` | Model identifier that produced the output, when known. |
| `media_type` | `str \| None` | Media type requested for the item, when provided. |
| `source` | `str \| None` | Sanitized source label; `None` for unnamed bytes/file-like inputs. |
| `warnings` | `tuple[str, ...]` | Extensible diagnostics channel; currently always empty. |

## Common arguments

| Argument | Type | Description |
| --- | --- | --- |
| `schema` | `type[BaseModel]` | Pydantic model class describing the desired output. |
| `model` | `str \| Model` | `pydantic-ai` model identifier or configured model instance. |
| `input_file` | `str \| os.PathLike[str] \| bytes \| BinaryIO \| ExtractionInput` | Local path, HTTP(S) URL, `Path`, bytes, binary file-like object, or `ExtractionInput`. |
| `instructions` | `str \| None` | Optional model guidance. |
| `style` | `ExtractionStyle \| str` | How the model inspects the input. `direct` (default) sends media in one shot. `search` gives the model sandboxed file tools (read/grep) against a text document. `code` lets the model write Python against a text document via [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/). `search` needs `pydantic-ai-harness`; `code` needs `pydantic-ai-harness[codemode]`. |
| `media_type` | `str \| None` | Required for bytes and file-like inputs without a per-item type; overrides inference for paths and URLs. Item-level `ExtractionInput.media_type` wins in batch calls. |
| `max_input_bytes` | `int \| None` | Per-input byte cap; `None` uses `OPENEXTRACT_MAX_INPUT_BYTES` or the 50 MiB default. |
| `max_retries` | `int` | Extra attempts after transient `ModelError`; defaults to `0`. |
| `retry_backoff` | `float` | Base seconds for exponential backoff with up to 25% jitter. |
| `retry_max_backoff` | `float` | Maximum delay, including provider `Retry-After`; defaults to `60.0`. |

Batch functions also accept:

| Argument | Type | Description |
| --- | --- | --- |
| `input_files` | `Iterable[str \| os.PathLike[str] \| bytes \| BinaryIO \| ExtractionInput]` | One input per extraction; items may carry their own media type. |
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

Session `model_settings` are passed directly to Pydantic AI using its typed
`ModelSettings` contract. `timeout` is an explicit shortcut for the model
request timeout and overrides a `timeout` entry in `model_settings`.
`url_timeout` separately controls URL input fetching. `instrument=True` enables
Pydantic AI instrumentation; an `InstrumentationSettings` instance provides
fine-grained control.

`Extractor` is bound to the thread that enters it and is not thread-safe.
`AsyncExtractor` is bound to one event loop; concurrent calls within that loop
are supported. Neither session can be reopened after it is closed.
