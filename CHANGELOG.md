# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]
- Add Anthropic provider support (extra `pydantic-ai-slim[anthropic]`; provider errors wrapped as `ModelError`).
- Add AWS Bedrock provider support (extra `pydantic-ai-slim[bedrock]`; `botocore.exceptions.ClientError` wrapped as `ModelError`).
- Add xAI (Grok) provider support (extra `pydantic-ai-slim[xai]`).
- Add Cohere provider support (extra `pydantic-ai-slim[cohere]`; `cohere.core.api_error.ApiError` wrapped as `ModelError`).
- Add Hugging Face provider support (extra `pydantic-ai-slim[huggingface]`; `huggingface_hub.errors.HfHubHTTPError` wrapped as `ModelError`).
- Add Groq provider support (extra `pydantic-ai-slim[groq]`; `groq.APIError` wrapped as `ModelError`).
- Add Cerebras provider support (model identifiers prefixed `cerebras:`, e.g. `cerebras:llama3.1-70b`). Uses the existing `openai` extra under the hood; set `CEREBRAS_API_KEY` in your environment.

## [0.5.0] - 2026-05-16
- Add `extract_async` for async extraction using `Agent.run`.
- Add `extract_many` and `extract_many_async` for concurrent batch extraction with configurable concurrency and optional exception capture.
- Accept raw `bytes` or any binary file-like object as `extract`'s `input_file`; new keyword-only `media_type` parameter for explicit MIME typing (required for `bytes`/file-like inputs, optional override for `str`).
- Add optional `max_retries` and `retry_backoff` keyword arguments to `extract()` for retrying transient `ModelError` failures with exponential backoff and jitter. Default behavior is unchanged (no retries).
- Add `extract_with_usage` and a `Usage` dataclass that surface model token counts (input, output, total) alongside the extracted output.
- Add `openextract` command-line interface (`openextract <file> --schema module:Class --model openai:gpt-5`) with structured exit codes.
- Add `examples/` directory with runnable scripts for invoice, receipt, and meeting-notes extraction.
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

[Unreleased]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.3.1...v0.3.2
[0.2.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.2...v0.1.4
[0.1.2]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Mellow-Artificial-Intelligence/openextract/releases/tag/v0.1.1
