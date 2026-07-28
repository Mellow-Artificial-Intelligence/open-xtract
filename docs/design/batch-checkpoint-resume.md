---
layout: page
title: Design — CLI batch checkpoint / resume
---

# Design: CLI batch checkpoint and resume

**Status:** proposal (not implemented)  
**Related issue:** [#154](https://github.com/Mellow-Artificial-Intelligence/openextract/issues/154)

## Goals

- Survive interrupted CLI batch jobs without redoing completed inputs.
- Preserve successful and failed item records.
- Keep stdout/stderr contracts in `docs/cli.md` intact (progress/checkpoint I/O
  must not corrupt stdout payloads).

## Non-goals (first cut)

- Checkpointing the Python `extract_many*` APIs (CLI-first).
- Deduplicating semantically identical remote URLs that change over time.
- Cross-host distributed workers.

## Proposed UX

```bash
openextract a.pdf b.pdf c.pdf \
  --schema pkg:Schema --model xai:grok-4.3 \
  --checkpoint .openextract/batch.jsonl \
  --output jsonl
```

Resume:

```bash
openextract a.pdf b.pdf c.pdf \
  --schema pkg:Schema --model xai:grok-4.3 \
  --checkpoint .openextract/batch.jsonl \
  --resume \
  --output jsonl
```

## On-disk format

JSONL sidecar (one object per completed input), separate from stdout:

```json
{"input":"a.pdf","status":"ok","result":{...},"input_fingerprint":"..."}
{"input":"b.pdf","status":"error","error":"...","error_type":"ModelError","input_fingerprint":"..."}
```

- **stdout** still receives the normal `--output json` array or `--output jsonl`
  stream for the current run's emissions (or a final merged view — pick one in
  implementation; prefer: resume run emits only newly completed lines on stdout
  when `--output jsonl`, and writes all completed records to the checkpoint).
- **stderr** keeps warnings/errors only.

Coordinate with #151 so checkpoint lines and stdout JSONL share the same
success/error object shapes where possible.

## Input identity

| Input kind | Fingerprint |
| ---------- | ----------- |
| Local path | `abspath` + `size` + `mtime_ns` (or content hash if `--checkpoint-hash`) |
| URL | Exact URL string (no auto-refresh) |
| stdin / bytes | Content sha256 |

Resume skips inputs whose fingerprint is already `ok` or `error` in the
checkpoint unless `--force-retry-errors` is set (errors only).

## Interaction with `--continue-on-error`

- Checkpoint writes after each item completes (success or error).
- Without `--continue-on-error`, abort still leaves prior items checkpointed.

## Failure modes / tests

- Corrupt checkpoint line → fail start with exit `1` and message.
- Missing file on `--resume` → exit `1`.
- Fingerprint mismatch (file changed) → re-run that item.
- Unit tests with mocked `extract_many` / per-item hooks; no live models.

## Deferred

- Python API checkpoint helpers.
- Automatic cloud object-store backends.
- Shrinking stdout to “delta only” for `--output json` (array) mode — JSONL is
  the natural resume companion.
