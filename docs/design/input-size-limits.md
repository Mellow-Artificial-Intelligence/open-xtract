---
layout: page
title: Design — input size limits
---

# Design: input size limits

Proposal for bounding how much data `openextract` loads before sending media to
a model. Today, local paths, URL responses, `bytes`, and file-like objects are
read fully into memory with no size cap.

**Status:** proposal (not implemented)  
**Related issue:** [#125](https://github.com/Mellow-Artificial-Intelligence/openextract/issues/125)

## Goals

- Prevent accidental OOM from huge local files or URL responses.
- Keep the public API small and predictable before 1.0.
- Fail with clear errors callers can catch.
- Preserve current behavior for normal document sizes.

## Non-goals (for the first implementation)

- Streaming media to providers.
- Per-modality compression / preprocessing.
- Guaranteeing provider-side upload limits (those remain upstream).

## Proposed defaults

| Knob | Default | Notes |
| ---- | ------- | ----- |
| Max input bytes | `52_428_800` (50 MiB) | Applies after bytes are resolved, before the model call |
| URL pre-check via `Content-Length` | enabled when header present | Missing/invalid header → stream/read with hard cap |
| Config surface | env + kwarg | Kwarg wins when both are set |

Suggested names:

- Env: `OPENEXTRACT_MAX_INPUT_BYTES`
- Kwarg on extract APIs: `max_input_bytes: int | None = None`
  (`None` means “use env/default”)

`0` or negative values should be rejected with `ValueError`.

## Behavior by input type

### Local paths

1. `Path.stat().st_size` when available.
2. If `st_size > limit` → raise before reading.
3. Otherwise `read_bytes()` and re-check `len(data)`.

### URLs

1. Existing SSRF / redirect / timeout controls unchanged.
2. If final response has a trustworthy `Content-Length` above the limit → fail
   before buffering the body.
3. If `Content-Length` is missing or wrong, read with a capped buffer / stream
   into a size-limited collector; exceed → fail.
4. Never trust `Content-Length` alone without also enforcing the cap on actual
   bytes received.

### Raw `bytes`

Reject when `len(input_file) > limit` before building the agent prompt.

### Binary file-like objects

Read through a capped reader (`read()` in chunks, stop at `limit + 1`). Do not
assume `.seek()` / `.tell()` exist.

### CLI

Inherit the same default and env var. Optional follow-up flag:
`--max-input-bytes N`. Stdin (`-`) uses the file-like path.

## Error type and messages

Add `InputTooLargeError(ExtractionError)`:

```text
Input exceeds the configured size limit (52428800 bytes); got at least 60000000 bytes.
Set OPENEXTRACT_MAX_INPUT_BYTES or pass max_input_bytes=... if this is intentional.
```

CLI mapping: new exit code **or** fold into `5` (`ExtractionError`) for the
first release to avoid expanding the provisional CLI contract. Prefer exit `5`
initially; document clearly.

## Backward compatibility

| Risk | Mitigation |
| ---- | ---------- |
| Existing large-file workflows start failing | Generous 50 MiB default; env/kwarg escape hatch |
| New exception type | Subclass of `ExtractionError`; broad `except ExtractionError` keeps working |
| CLI exit codes | Map to existing `5` first; consider a dedicated code only if needed later |

This is a behavior change for inputs larger than the default, so it should ship
in a minor pre-1.0 release and be called out as breaking in `CHANGELOG.md`.

## Security benefits

- Bounds memory growth from huge local files and hostile/misconfigured URLs.
- Complements SSRF controls (which do not limit response body size today).
- Reduces accidental cost from uploading enormous payloads to paid model APIs.

## Smallest safe implementation

1. Shared helper `_enforce_max_input_bytes(data: bytes, *, limit: int) -> bytes`.
2. Apply in `_get_media` after resolution (and URL `Content-Length` fast-fail in
   `_fetch_url` / `_read_from_path`).
3. Thread `max_input_bytes` through sync/async/batch/usage APIs.
4. Unit tests: local oversize, bytes oversize, URL with large `Content-Length`,
   URL with missing header + oversized body, env override, kwarg override.
5. Docs: README + troubleshooting entry.

## Follow-up implementation issues (when accepted)

1. Implement `InputTooLargeError` + 50 MiB default + env/kwarg.
2. Cap URL body reads when `Content-Length` is absent.
3. Optional CLI `--max-input-bytes`.
4. Evaluate whether a dedicated CLI exit code is warranted after the Python API
   lands.
