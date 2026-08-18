# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Breaking changes are called out explicitly in the relevant release section.
Deprecations should be listed with the replacement path and expected removal
timing when that is known.

## [Unreleased]

### Added
- Swarms: `extract_swarm` / `extract_swarm_async` run several agents over one
  input and return the reduced result, and
  `extract_swarm_with_results*` additionally report each agent's
  `ExtractionResult`, the summed usage, and the reduce strategy. Agents are a
  model identifier, a configured pydantic-ai `Model`, or a `SwarmMember` with
  per-agent `instructions` / `style`; `size` fans one agent out up to 16 ways.
  The input is loaded once for the whole swarm.
- Swarm reduce strategies: `SwarmReduce` (`merge`, `vote`, `first`),
  `normalize_reduce`, and `reduce_outputs` fold several same-schema outputs
  into one validated instance.
- `scripts/extractbench.py` runs [ExtractBench](https://github.com/run-llama/ExtractBench)
  through openextract with any `pydantic-ai` model identifier (`--model openai:gpt-5 --test`).
- Extraction styles: `style='direct'` (default) still sends media to the model
  in one shot; `style='search'` uses Pydantic AI Harness `FileSystem` tools
  (read, regex search, glob) on text documents; `style='code'` uses Harness
  `CodeMode` so the model can write Python against a workspace copy of the
  text. Available on every extract API, reusable sessions, and as `--style`
  on the CLI. Install `pydantic-ai-harness` for search and
  `pydantic-ai-harness[codemode]` for code execution.
- Typed input and result contracts: `ExtractionInput` supports direct
  `os.PathLike` sources with per-item `media_type` and an optional safe
  `name`, and `ExtractionResult[T]` carries output, usage, attempts, timing,
  model/media metadata, and warnings without retaining raw media or secrets.
- `Path`/`os.PathLike` now works directly in every public API, and batch calls
  accept heterogeneous inputs with per-item media types in a single run.
- `extract_many_with_results` / `extract_many_with_results_async` return
  per-item `ExtractionResult` diagnostics, and `total_usage` aggregates token
  usage across batch results.
- `Literal[True/False]` overloads on `extract_many*` and
  `iter_extract_many_async` so type checkers infer `list[T]` versus
  `list[T | Exception]` from `return_exceptions`.
- Reusable `Extractor` and `AsyncExtractor` sessions with deterministic agent
  and HTTP-client cleanup, configured Pydantic AI model or agent injection,
  model settings/timeouts/instrumentation, and typed `RetryPolicy` support.
- `InputTooLargeError`, a 50 MiB default per-input cap, the
  `OPENEXTRACT_MAX_INPUT_BYTES` environment variable, `max_input_bytes` on all
  extraction APIs, and `--max-input-bytes` on the CLI.
- Drift-checked canonical API reference, supported-Python CI matrix, and wheel
  and source-distribution install smoke tests.
- `iter_extract_many_async()` streams `(input_index, result)` pairs in
  completion order without waiting for the full batch.
- Streaming-batch example and docs comparing `iter_extract_many_async`
  (completion order) with `extract_many` (input order).
- GitHub Pages now builds a documentation site: user [guide](docs/guide.md),
  [agent contract](docs/agents.md), rendered API/CLI/provider pages, and
  [`llms.txt`](docs/llms.txt).

### Changed
- Split the internal extraction implementation into focused modules (`_types`,
  `_config`, `_media`, `_errors`, `_retry`, `_agent`, `_session`, `_batch`) so
  input loading, retries, sessions, and batch execution are no longer one file.
  Public APIs are unchanged.
- CI skips jobs that the change set cannot affect, cancels outdated pull-request
  runs, caches uv downloads, and collects coverage on Python 3.12 only. Releases
  publish only after CI succeeds on `main`, and docs deploy only when `docs/`
  changes.
- `openai:` model identifiers now use the OpenAI Responses API by default;
  `openai-chat:` remains available as an explicit Chat Completions opt-in.
- Batch execution now schedules at most `max_concurrency` inputs, consumes
  generator inputs lazily, and cancels and awaits outstanding work on fail-fast
  errors while preserving input order in the existing list APIs.
- Python API calls no longer load `.env` into process-wide state. The CLI and
  bundled examples continue to load `.env` explicitly.
- Examples run as `python -m examples.<module>` without mutating `sys.path`.
- Package import defers Pydantic AI runtime modules, and model-error
  classification inspects the raised exception's MRO without importing
  unrelated provider SDKs.

### Security
- Paths, URLs, bytes, file-like objects, stdin, and batch inputs now fail before
  a model call when they exceed the configured cap. URL bodies are streamed and
  bounded even when `Content-Length` is missing or incorrect.

## [0.10.0] - 2026-08-01

### Added
- Explicit `RuntimeError` when `extract_many()` is called from a running event
  loop, directing callers to `extract_many_async`.
- Maintainer docs: CLI stdout/stderr/exit-code contracts, provider capability
  matrix, troubleshooting guide, live smoke harness notes, release checklist,
  and an input-size-limits design proposal.
- Opt-in live provider smoke test (`OPENEXTRACT_LIVE_SMOKE=1`) for a
  representative OpenAI image path.
- Expanded `SECURITY.md` URL input security model (schemes, host validation,
  redirects, env configuration, and non-guarantees).

### Changed
- Async extraction now keeps disk, DNS, and file-like reads off the event loop
  and reuses one HTTP client across each batch.
- Model retries now reuse the original media payload, prompt, and agent instead
  of reading or fetching the input and rebuilding the agent on every attempt.
- Model retries now distinguish transient failures from permanent provider
  errors, honor bounded `Retry-After` values, and expose provider, status,
  retryability, and retry-after metadata on `ModelError`.
- Added `retry_max_backoff` to all extraction APIs and
  `--retry-max-backoff` to the CLI.

### Security
- Async URL fetching retains redirect-by-redirect public-host and SSRF
  validation while using async clients and offloaded DNS resolution.

## [0.9.0] - 2026-07-12

### Added
- Compatibility and deprecation policy documenting the public API surface,
  pre-1.0 breaking-change expectations, provider compatibility limits, and
  Python version support policy.
- `ProviderNotInstalledError` (a subclass of `ExtractionError`) raised with an
  actionable `pip install openextract[...]` hint when a model is requested whose
  provider extra is not installed. The CLI reports it with exit code `6`.
- `--continue-on-error` flag on the `openextract` CLI: in batch mode, keep
  processing remaining inputs when one fails, emit per-item errors inline, and
  exit `7` if any input failed (default remains abort-on-first-failure).
- README public API stability audit covering every symbol exported from
  `openextract.__all__`, plus CLI stability notes and pre-1.0 follow-ups.
- Release-readiness documentation for provider install errors, extra-specific
  install hints, and CLI partial-batch failure stdout/stderr and exit-code
  behavior.

### Changed
- `max_retries`, `retry_backoff`, and `max_concurrency` now fail early with
  deterministic `ValueError` messages when invalid values are provided.

## [0.8.0] - 2026-06-02

### Added
- Environment variables `OPENEXTRACT_URL_TIMEOUT` and `OPENEXTRACT_MAX_REDIRECTS`
  to configure HTTP timeout and redirect limits when fetching URLs.
- Ship `py.typed` for PEP 561 type checker support.
- Run [ty](https://docs.astral.sh/ty/) on `src/openextract` in CI (Astral toolchain with uv and ruff).
- `max_retries` and `retry_backoff` on `extract_async`, `extract_with_usage`,
  `extract_with_usage_async`, `extract_many`, and `extract_many_async` (per-item
  for batch), matching sync `extract()` retry semantics.
- Optional dependency extras: `openai`, `anthropic`, `google`, `bedrock`,
  `cohere`, `groq`, `huggingface`, `mistral`, `openrouter`, `xai`, `logfire`,
  and `all`.
- Examples `batch_invoices.py` and `extract_with_usage.py` for concurrent batch
  extraction and token usage reporting.
- CLI accepts multiple `input_file` arguments for batch extraction via `extract_many`.
- `--usage`, `--media-type`, and stdin (`-`) support on the `openextract` command.

### Changed
- **Breaking:** `pip install openextract` no longer bundles every provider SDK.
  Install a provider extra for model calls (e.g. `openextract[openai]` or `openextract[all]`).
- Expand README API reference to document `extract_async`, `extract_many`,
  `extract_with_usage`, and related public APIs.

## [0.7.0] - 2026-05-22
### Added
- Add `extract_with_usage_async`: async counterpart to `extract_with_usage` that
  returns `(output, Usage)` alongside the token counts for the model call.
- Add `media_type` keyword argument to `extract_many` and `extract_many_async`:
  the batch functions previously hard-coded `None`, making it impossible to process
  `bytes` or file-like inputs that require an explicit MIME type; the new parameter
  is applied uniformly to every item in the batch.
- Add `--max-retries` and `--retry-backoff` flags to the `openextract` CLI:
  exposes the existing retry logic (already available via the Python API) to
  command-line users.

### Security
- URL fetching now refuses hosts that resolve to non-public addresses
  (private/loopback/link-local/multicast/reserved, IPv4 and IPv6, including
  IPv4-mapped IPv6 and the `169.254.169.254` cloud-metadata endpoint) to
  reduce SSRF risk when callers pass untrusted URLs. The check is re-applied
  at every redirect hop; set `OPENEXTRACT_ALLOW_PRIVATE_URLS=1` to opt out.
  **Breaking** for callers that previously fetched `localhost`/internal
  hosts via `extract()`.
- Pinned all GitHub Actions to commit SHAs.

## [0.6.0] - 2026-05-16
- Add Anthropic provider support (extra `pydantic-ai-slim[anthropic]`; `anthropic.APIError` wrapped as `ModelError`).
- Add AWS Bedrock provider support (extra `pydantic-ai-slim[bedrock]`; `botocore.exceptions.ClientError` wrapped as `ModelError`).
- Add xAI (Grok) provider support (extra `pydantic-ai-slim[xai]`).
- Add Cohere provider support (extra `pydantic-ai-slim[cohere]`; `cohere.core.api_error.ApiError` wrapped as `ModelError`).
- Add Hugging Face provider support (extra `pydantic-ai-slim[huggingface]`; `huggingface_hub.errors.HfHubHTTPError` wrapped as `ModelError`).
- Add Groq provider support (extra `pydantic-ai-slim[groq]`; `groq.APIError` wrapped as `ModelError`).
- Add Mistral provider support (extra `pydantic-ai-slim[mistral]`; `mistralai.client.errors.mistralerror.MistralError` wrapped as `ModelError`).
- Add OpenRouter provider support (extra `pydantic-ai-slim[openrouter]`; openai-compatible, so the existing `openai.APIError` classification applies).
- Document Cerebras provider support (`cerebras:` prefix; openai-compatible — no dedicated extra needed).
- Document Outlines provider support (`outlines:` prefix; install separately with a backend extra such as `pydantic-ai-slim[outlines-transformers]`).
- Clarify Ollama support: it works via the `openai`-compatible code path; `pydantic-ai-slim` does not publish a dedicated `ollama` extra, so no separate dependency is required.

## [0.5.0] - 2026-05-16
- Add `extract_async` for async extraction using `Agent.run`.
- Add `extract_many` and `extract_many_async` for concurrent batch extraction with configurable concurrency and optional exception capture.
- Accept raw `bytes` or any binary file-like object as `extract`'s `input_file`; new keyword-only `media_type` parameter for explicit MIME typing (required for `bytes`/file-like inputs, optional override for `str`).
- Add optional `max_retries` and `retry_backoff` keyword arguments to `extract()` for retrying transient `ModelError` failures with exponential backoff and jitter. Default behavior is unchanged (no retries).
- Add `extract_with_usage` and a `Usage` dataclass that surface model token counts (input, output, total) alongside the extracted output.
- Add `openextract` command-line interface (`openextract <file> --schema module:Class --model openai:gpt-5`) with structured exit codes.
- Add `examples/` directory with runnable scripts for invoice, receipt, and meeting-notes extraction.
- Reorganize `examples/` into use-case folders (basic, images, documents, batch, async, advanced, audio, CLI), add bundled fixtures, `run_all.py`, and smoke tests.
- Replace the substring-based exception classifier in `extract()` with typed provider-error matching against `pydantic_ai.exceptions.ModelAPIError`, `openai.APIError`, and `google.genai.errors.APIError`. Behavior change: an arbitrary exception whose message merely mentions "model" is no longer promoted to `ModelError`; it is now wrapped as `ExtractionError` unless the exception type is a subclass of a known provider error.

## [0.4.0] - 2026-05-16
- Accept `http://` URLs in addition to `https://`; previously, plain-HTTP URLs were silently treated as local file paths.
- Raise `UrlFetchError` on non-2xx responses; previously, the HTML error body was passed to the LLM as media bytes.
- Follow HTTP redirects when fetching URLs and apply a 30-second timeout.
- Fall back to the response `Content-Type` header when the URL has no recognizable extension (e.g., `/download?id=42`).
- Reach 100% test coverage and enforce the threshold in CI.
- Remove `configure_logging` from `__all__` — it was never defined, breaking `from openextract import *`.
- Fix `extract()` docstring (`url` → `input_file`) and add type hints to `_get_media`.

## [0.3.2] - 2026-05-05
- Add Ollama model support.

## [0.2.0] - 2026-01-11
- Landing page redesign and security updates.

## [0.1.4] - 2025-12-21
- Restructure project as installable Python package.
- Add tests and error handling.
- Initial commit: media extraction utility with pydantic-ai.

## [0.1.2] - 2025-09-13
- Add bytes-only vision API.
- Render PDFs to images.
- Support multimodal messaging.

## [0.1.1] - 2025-09-10
- Merge pull request #12 from Mellow-Artificial-Intelligence/new-release.

[Unreleased]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.3.1...v0.3.2
[0.2.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.2...v0.1.4
[0.1.2]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Mellow-Artificial-Intelligence/openextract/releases/tag/v0.1.1
