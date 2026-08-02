---
layout: page
title: Design — input size limits
---

# Design: input size limits

Implementation notes for bounding how much data `openextract` loads before
sending media to a model.

**Status:** implemented
**Related issue:** [#163](https://github.com/Mellow-Artificial-Intelligence/openextract/issues/163)

## Goals

- Prevent accidental OOM from huge local files or URL responses.
- Keep the public API small and predictable before 1.0.
- Fail with clear errors callers can catch.
- Preserve current behavior for normal document sizes.

## Non-goals (for the first implementation)

- Streaming media to providers.
- Per-modality compression / preprocessing.
- Guaranteeing provider-side upload limits (those remain upstream).

## Defaults

| Knob | Default | Notes |
| ---- | ------- | ----- |
| Max input bytes | `52_428_800` (50 MiB) | Applies after bytes are resolved, before the model call |
| URL pre-check via `Content-Length` | enabled when header present | Missing/invalid header → stream/read with hard cap |
| Config surface | env + kwarg | Kwarg wins when both are set |

Configuration names:

- Env: `OPENEXTRACT_MAX_INPUT_BYTES`
- Kwarg on extract APIs: `max_input_bytes: int | None = None`
  (`None` means “use env/default”)

`0` or negative values are rejected with `ValueError`.

## Behavior by input type

### Local paths

1. `Path.stat().st_size` when available.
2. If `st_size > limit` → raise before reading.
3. Otherwise read in bounded chunks and enforce the final byte count.

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

The CLI inherits the same default and environment variable and exposes
`--max-input-bytes N`. Stdin (`-`) is passed through the bounded file-like path
instead of being read eagerly by argument handling.

## Error type and messages

`InputTooLargeError(ExtractionError)` reports the safe source context and limit:

```text
Input exceeds the configured size limit (52428800 bytes); got at least 60000000 bytes.
Set OPENEXTRACT_MAX_INPUT_BYTES or pass max_input_bytes=... if this is intentional.
```

The CLI maps this error to the existing `5` (`ExtractionError`) exit code.

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

## Implementation

1. `_resolve_max_input_bytes` applies explicit-call, environment, and default
   precedence with positive-integer validation.
2. Local paths use `stat()` before opening and a capped chunk reader afterward.
3. URLs use streaming HTTP responses, `Content-Length` fast-fail, and an actual
   byte-count cap on every redirect's final response.
4. Raw bytes and non-seekable binary streams share the same cap.
5. Batch preparation remains inside the concurrency semaphore, bounding
   simultaneous buffered payloads to `max_concurrency`.
