---
layout: page
title: Troubleshooting
---

# Troubleshooting

Short recovery steps for common setup and runtime failures. Exception names and
CLI exit codes match `src/openextract/_cli.py` and `src/openextract/exceptions.py`.

## Missing provider SDK extras

**Symptoms**

- Python: `ProviderNotInstalledError`
- CLI: stderr `error: ...`, exit code `6`

**Cause**

The base package does not install provider SDKs. Calling a model whose extra is
missing raises this error.

**Next step**

Install the hinted extra, for example:

```bash
pip install 'openextract[openai]'
# or
pip install 'openextract[xai]'
# or every provider
pip install 'openextract[all]'
```

## Missing provider credentials

**Symptoms**

- Python: `ModelError` (or a provider SDK auth error wrapped as `ModelError`)
- CLI: exit code `4`

**Cause**

The provider SDK is installed, but no API key / cloud credentials are available.

**Next step**

Set the provider env var (loaded automatically from `.env`), for example
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`. See the
[provider matrix](providers.md) for the credential column.

## URL fetch failures

**Symptoms**

- Python: `UrlFetchError`
- CLI: exit code `2`

**Common causes**

- Host refused as non-public (SSRF protection)
- HTTP error / timeout / too many redirects
- DNS failure

**Next step**

1. Confirm the URL is `http://` or `https://` and publicly reachable.
2. For intentional localhost/internal fetches, either set
   `OPENEXTRACT_ALLOW_PRIVATE_URLS=1` (understand the risk) or fetch bytes
   yourself and pass `bytes` / a file-like object with `media_type`.
3. Tune `OPENEXTRACT_URL_TIMEOUT` / `OPENEXTRACT_MAX_REDIRECTS` if needed.
4. Read [SECURITY.md](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/SECURITY.md#url-input-security-model).

## Input too large

**Symptoms**

- Python: `InputTooLargeError`
- CLI: exit code `5`

**Cause**

Resolved input (local file, URL body, `bytes`, or file-like) exceeds the
configured size limit (default 50 MiB).

**Next step**

1. Confirm the file/URL is the intended input and not accidentally huge.
2. Raise the limit when intentional:

```bash
export OPENEXTRACT_MAX_INPUT_BYTES=104857600  # 100 MiB
```

```python
extract(..., max_input_bytes=100_000_000)
```

## Schema validation failures

**Symptoms**

- Python: `SchemaValidationError`
- CLI: exit code `3`

**Cause**

The model returned output that could not be validated against your Pydantic
schema.

**Next step**

- Tighten `instructions` and keep the schema fields extractable from the media.
- Prefer capable models for the modality (for example vision for images).
- Retry with `max_retries` if the failure is intermittent model drift.

## Model API failures and retries

**Symptoms**

- Python: `ModelError`
- CLI: exit code `4`

**Cause**

Provider/model API failure (auth, rate limit, server error, unsupported media).

**Next step**

```python
extract(..., max_retries=3, retry_backoff=1.0)
```

CLI:

```bash
openextract ./file.pdf --schema pkg:Model --model openai:gpt-5 \
  --max-retries 3 --retry-backoff 1.0
```

Retries apply only to `ModelError`. Invalid option values raise `ValueError`
(CLI exit `1`).

## CLI schema import errors

**Symptoms**

- CLI: stderr `error: ...`, exit code `1`

**Cause**

`--schema` must be `module:ClassName` pointing at a Pydantic `BaseModel`
subclass importable from the current `PYTHONPATH`.

**Next step**

```bash
# from the repo root, for the bundled example schema
PYTHONPATH=. openextract ./file.png \
  --schema examples.cli.schemas:DocumentInfo \
  --model xai:grok-4.3
```

Confirm the module imports cleanly: `python -c "import examples.cli.schemas"`.

## Batch partial failures

**Symptoms**

- CLI with `--continue-on-error`: exit code `7`, stderr warning, stdout still has
  the full JSON array
- Python `extract_many(..., return_exceptions=True)`: exceptions appear in-place

**Next step**

Inspect per-item `error` / `error_type` entries. Fix the failing inputs, or omit
`--continue-on-error` / `return_exceptions` to fail fast on the first error.

## Sync batch inside an async event loop

**Symptoms**

- `RuntimeError: extract_many() cannot be called from a running event loop...`

**Next step**

Use the async API:

```python
results = await extract_many_async(schema=..., model=..., input_files=...)
```

## Related

- [CLI contracts](cli.md)
- [Provider matrix](providers.md)
- [1.0 freeze checklist](design/1.0-freeze.md)
- [README error handling](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/README.md#error-handling)
