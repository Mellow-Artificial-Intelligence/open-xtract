---
layout: page
title: Design — rate limiting
---

# Design: rate limiting separate from max_concurrency

**Status:** proposal (not implemented)  
**Related issue:** [#153](https://github.com/Mellow-Artificial-Intelligence/openextract/issues/153)

## Problem

`max_concurrency` bounds in-flight work but does not pace requests over time.
Provider quotas are often expressed as requests/minute (or tokens/minute). A
burst of concurrent calls can still 429 even with a low semaphore.

## Smallest useful API

```python
extract_many(
    ...,
    max_concurrency=5,
    max_requests_per_minute: int | None = None,
)
```

Semantics:

- `None` (default): no pacing beyond concurrency (today's behavior).
- Positive int: minimum spacing ≈ `60 / max_requests_per_minute` between
  **starts** of model calls (not including retry waits already governed by
  `retry_backoff`).
- Invalid values (`<= 0`, bools): `ValueError`, same style as
  `max_concurrency`.

Token-budget limiting is **deferred** (providers report tokens after the call;
pre-flight estimates are unreliable).

## Interaction with retries

- Each attempt (including retries) counts as a request toward the rate limit.
- Retry backoff and rate-limit spacing both apply; do not disable one when the
  other is set.
- `ModelError` retries remain as today; a 429 classified as `ModelError` will
  back off and also wait for the rate limiter before the next start.

## Implementation sketch

- Shared async rate gate used inside `_gather_extractions` around each attempt.
- Use `time.monotonic()` (injectable clock for tests).
- Sync `extract_many` inherits via `asyncio.run` as today.

## CLI

Optional follow-up: `--max-requests-per-minute N` on batch runs only. Not
required for the first Python API cut.

## Privacy / logging

- Do not log prompt contents when rate limiting.
- Optional debug log: “rate limiter waited X ms” without media/secrets.

## Test strategy

- Mock monotonic clock / sleep to assert spacing without wall-clock waits.
- Concurrency + rate limit together: never exceed N starts per simulated minute.
- Invalid option validation unit tests.

## Deferred to implementation

- Token/minute budgets.
- Adaptive throttling from provider response headers.
- Global process-wide limiter across multiple `extract_many` calls.
