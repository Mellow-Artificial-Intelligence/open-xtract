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
| `1` | Usage / setup error | Bad `--schema`, stdin without `--media-type`, `--usage` with multiple inputs, invalid retry options, argparse failures |
| `2` | URL fetch error | `UrlFetchError` (network failure, HTTP error, SSRF refusal) |
| `3` | Schema validation error | `SchemaValidationError` |
| `4` | Model API error | `ModelError` |
| `5` | Other extraction error | Other `ExtractionError` subclasses (including `InputTooLargeError`) |
| `6` | Missing provider SDK | `ProviderNotInstalledError` |
| `7` | Partial batch failure | `--continue-on-error` with one or more per-item failures |

These mappings live in `src/openextract/_cli.py` and are covered by
`tests/test_cli.py`.

Oversized inputs raise `InputTooLargeError` and exit `5`. Tune the limit with
`OPENEXTRACT_MAX_INPUT_BYTES` (default 50 MiB).

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

## `--output json`, `--output jsonl`, and `--output repr`

| Flag | Behavior |
| ---- | -------- |
| `--output json` (default) | Pretty-printed JSON (`indent=2`). Single-file non-usage results use `model_dump_json`. Batch mode emits a JSON array. |
| `--output jsonl` | Batch only: one JSON object per line, in input order. Rejected for single-file / `--usage` (exit `1`). |
| `--output repr` | Python `repr(...)` of the same payload object. |

Both JSON formats write only to stdout on success. JSONL with
`--continue-on-error` emits the same per-item error objects as the array form,
one line each; stderr still gets the partial-failure warning and exit `7`.

## Stdin input

```bash
cat ./reports/q4.pdf | openextract - \
  --schema mypkg.schemas:Invoice \
  --model xai:grok-4.3 \
  --media-type application/pdf
```

- `-` reads raw bytes from stdin.
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

## Related docs

- [Troubleshooting](troubleshooting.md)
- [URL security model](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/SECURITY.md#url-input-security-model)
