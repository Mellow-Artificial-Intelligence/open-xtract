---
layout: page
title: CLI contracts
---

# CLI stdout, stderr, and exit codes

This page is the contract for the `openextract` command. Successful results go
to **stdout**. Errors and warnings go to **stderr**. Exit codes are stable for
automation.

## Streams

| Stream | Contents |
| ------ | -------- |
| stdout | Successful extraction payloads (`json` or `repr`) |
| stderr | `error: ...` messages, plus a `warning: ...` line for partial batch failures |

Never parse stderr for successful results. Never treat stdout as empty when
exit code `7` is returned — the batch array is still written.

## Exit codes

| Code | Meaning | Typical cause |
| ---- | ------- | ------------- |
| `0` | Success | Single-file or full-batch success |
| `1` | Usage / setup error | Bad `--schema`, stdin without `--media-type`, `--usage` with multiple inputs, invalid retry/size options, argparse failures |
| `2` | URL fetch error | `UrlFetchError` (network failure, HTTP error, SSRF refusal) |
| `3` | Schema validation error | `SchemaValidationError` |
| `4` | Model API error | `ModelError` |
| `5` | Other extraction error | `InputTooLargeError` and other `ExtractionError` subclasses |
| `6` | Missing provider SDK | `ProviderNotInstalledError` |
| `7` | Partial batch failure | `--continue-on-error` with one or more per-item failures |

These mappings live in `src/openextract/_cli.py` and are covered by
`tests/test_cli.py`.

## Successful single-file output

```bash
openextract ./reports/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --output json
```

- Exit `0`.
- stdout: JSON object from `model_dump_json` (default `--output json`).
- With `--output repr`, stdout is `repr(payload)` instead of JSON.

## Successful batch output

```bash
openextract ./invoices/a.pdf ./invoices/b.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3
```

- Exit `0` when every item succeeds.
- stdout: JSON array of per-item `model_dump()` objects, in input order.

## `--usage` output

```bash
openextract ./reports/q4.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --usage
```

- Single input only; multiple inputs exit `1` with an error on stderr.
- stdout JSON shape:

```json
{
  "result": { "...": "schema fields" },
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
  }
}
```

## `--output json` and `--output repr`

| Flag | Behavior |
| ---- | -------- |
| `--output json` (default) | Pretty-printed JSON (`indent=2`). Single-file non-usage results use `model_dump_json`. |
| `--output repr` | Python `repr(...)` of the same payload object. |

Both formats write only to stdout on success.

## Stdin input

```bash
cat ./reports/q4.pdf | openextract - \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --media-type application/pdf
```

- `-` reads raw bytes from stdin.
- Stdin is read in bounded chunks under the same input-size limit as files and URLs.
- `--media-type` is required.
- `-` cannot be combined with other input paths (exit `1`).

## `--continue-on-error` partial failures

```bash
openextract ./ok.pdf ./missing.pdf \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --continue-on-error
```

- stdout still receives the full JSON array.
- Failed items appear as:

```json
{
  "input": "./missing.pdf",
  "error": "...",
  "error_type": "ModelError"
}
```

- stderr receives a warning: `warning: N of M input(s) failed; see output for details`.
- Exit `7` when any item failed; exit `0` when none failed.
- Without `--continue-on-error`, the first failure aborts and maps to exit codes
  `2`–`6` / `1` as usual (no batch array).

## Retry policy

`--max-retries` enables retries for transient model failures only.
`--retry-backoff` controls exponential backoff with up to 25% additive jitter,
and `--retry-max-backoff` caps both calculated delays and provider
`Retry-After` values. Authentication, permission, and invalid-request failures
exit immediately with code `4`.

## Input size limit

`--max-input-bytes N` sets the maximum bytes loaded for each input. Without the
flag, the CLI uses `OPENEXTRACT_MAX_INPUT_BYTES` or the 50 MiB default. Values
must be positive integers. Oversized inputs fail before a model call and exit
`5`; URL bodies and stdin remain bounded even without a reliable length header.

## Extraction styles

`--style direct` (default) sends the resolved media to the model in one shot.
`--style search` and `--style code` are text-only: search gives the model
sandboxed file tools (read, regex search, glob), and code lets it write Python
against a workspace copy of the document via Pydantic AI Harness. Missing extras
exit `6` (`ProviderNotInstalledError`). Non-text inputs raise `ValueError`
(exit `1`).

## Related docs

- [Troubleshooting](troubleshooting.md)
- [URL security model](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/SECURITY.md#url-input-security-model)
