---
layout: page
title: Design — ExtractResult / diagnostics
---

# Design: ExtractResult / richer diagnostics API

**Status:** proposal (not implemented)  
**Related issue:** [#149](https://github.com/Mellow-Artificial-Intelligence/openextract/issues/149)

## Goals

- Give callers usage and light diagnostics without breaking `extract() -> T`.
- Define a batch diagnostics shape that does not fight future rate-limit /
  checkpoint work.
- Keep secrets and raw media out of logs and returned diagnostics.

## Non-goals (1.0 / first cut)

- Replacing `extract_with_usage*` (those stay).
- Full tracing / OpenTelemetry exporters.
- Persisting diagnostics to disk (see checkpoint design).

## Recommendation (smallest useful version)

**Keep `extract()` returning only `T`.**

Add an opt-in result wrapper later as:

```python
@dataclass(frozen=True)
class ExtractResult(Generic[T]):
    output: T
    usage: Usage
    model: str
    media_type: str
    # deferred: duration_ms, retry_count, warnings
```

Preferred public entry points:

- `extract_result(...)` / `extract_result_async(...)` returning `ExtractResult[T]`
- **or** document that callers who only need tokens should keep using
  `extract_with_usage*`

Do **not** change `extract()`'s return type.

### Batch

Prefer **new** helpers over changing `extract_many*` defaults:

- `extract_many_with_usage(...)` → `list[tuple[T | BaseException, Usage | None]]`
  plus a documented aggregate helper, **or**
- `extract_many_result(...)` → `list[ExtractResult[T] | BaseException]`

Failed items under `return_exceptions=True`:

- Exception in-place (existing behavior)
- Usage: `None` when the model was never called; partial usage only if a
  successful attempt existed then a later failure (unlikely with current retry
  model — document as `None` on failure for v1)

Coordinate with #152 so batch usage and ExtractResult do not ship two competing
shapes. Prefer one batch diagnostics design.

### Fields

| Field | v1 | Later |
| ----- | -- | ----- |
| `output` | yes | |
| `usage` | yes | |
| `model` | yes (echo request model id) | |
| `media_type` | yes (resolved) | |
| `source` | no | path/URL redacted form |
| `provider` | no | parsed prefix |
| `duration_ms` | no | |
| `retry_count` | no | |
| `warnings` | no | |

## Privacy

- Never include raw media bytes, file contents, or API keys in `ExtractResult`,
  logs, or CLI output beyond existing error messages.
- URL/`input` echoes in CLI batch errors may remain paths/URLs as today; do not
  expand to bodies.

## Backward compatibility

- Additive APIs only; `extract()` unchanged.
- New exports go in `__all__` as Provisional until proven.

## Tests / docs required for implementation

- Unit tests for success, validation failure (no usage), model failure after
  retries, and batch mixed success/failure.
- README API reference + stability table rows.
- Changelog entry under Added.

## Deferred to implementation issues

File implementation only after maintainer acceptance of this note. Likely split:

1. `ExtractResult` + `extract_result*` (single-item).
2. Batch usage / result helpers (#152).
