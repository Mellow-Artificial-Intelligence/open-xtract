# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]
- Add `openextract` command-line interface for running extractions from the shell.
- Add `examples/` directory with runnable scripts for invoice, receipt, and meeting-notes extraction.
- Replace the substring-based exception classifier in `extract()` with typed provider-error matching against a tuple of known model-error base classes (`pydantic_ai.exceptions.ModelAPIError`, `openai.APIError`, `google.genai.errors.APIError`). Behavior change: an arbitrary exception whose message merely mentions "model" is no longer promoted to `ModelError`; it is now wrapped as `ExtractionError` unless the exception type is a subclass of a known provider error.
- Accept raw `bytes` or any binary file-like object as `extract`'s `input_file`.
- Add keyword-only `media_type` parameter for explicit MIME typing; required for `bytes` and file-like inputs, and overrides the guess for `str` paths and URLs.
- Add `extract_async` for async extraction using `Agent.run`.
- Add `extract_many` and `extract_many_async` for concurrent batch extraction with configurable concurrency and optional exception capture.

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

[Unreleased]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.3.1...v0.3.2
[0.2.0]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.2...v0.1.4
[0.1.2]: https://github.com/Mellow-Artificial-Intelligence/openextract/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Mellow-Artificial-Intelligence/openextract/releases/tag/v0.1.1
