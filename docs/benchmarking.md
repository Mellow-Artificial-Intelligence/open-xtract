---
layout: page
title: Benchmarking
---

# Benchmarking (maintainer tool)

`scripts/bench.py` is a maintainer-facing microbenchmark for openextract's
**local** hot path — the CPU work that runs around every extraction call. It
exists to catch local performance regressions during development. It is **not**
a published performance benchmark, and its numbers are not portable across
machines.

## How to run

From the repository root, with the [uv](https://docs.astral.sh/uv/)
environment set up (`uv sync --dev`):

```bash
uv run python scripts/bench.py
```

It needs no API keys and makes no network calls — it stubs dummy provider
credentials and mocks the LLM call. Results are printed to stdout and the
script exits `0`.

## What it measures

Most rows report `median`, `p95`, and `best` over `n` iterations (after a
warmup), so both typical and tail cost are visible. The `[import]` row instead
reports `median`, `best`, and `worst` over a fixed 5 runs:

- **`[import]`** — cold-start cost of `import openextract`, measured in a fresh
  subprocess (5 runs).
- **`[_get_media]`** — resolving an `input_file` to `(bytes, media_type)` for a
  small text file, a ~50 KB PDF, a ~5 MB PDF, and raw `bytes` input.
- **`[_get_media_type]`** — `mimetypes.guess_type` on a filename.
- **`[dotenv]`** — per-call cost of `load_dotenv()` (invoked inside every
  extract).
- **`[_build_agent]`** — constructing a `pydantic_ai.Agent` for a non-Ollama
  and an Ollama model.
- **`[extract]` / `[extract_many]`** — the full `extract()` / `extract_many()`
  path with the `Agent` **mocked out**: all per-call local cost (dotenv + media
  read + agent build + dispatch) *except* the model call.

## What it intentionally does NOT measure

- **Model / network / inference latency.** The LLM round-trip is mocked. In
  real usage it dominates total time by orders of magnitude; this tool
  deliberately excludes it so the local CPU costs are visible at all.
- **Provider SDK network calls**, rate limits, retries, or token costs.
- **Cross-machine or absolute performance.** Numbers reflect the specific
  machine, Python build, and system load at the moment of the run.

## When maintainers should run it

Run it **before and after** a change that touches the local hot path, and
compare the two runs on the **same machine**:

- changes to `import openextract` or module-level import work,
- `_get_media` / media handling,
- `_build_agent` / agent construction,
- the per-call `load_dotenv()` behavior,
- `extract` / `extract_many` dispatch and concurrency.

It is also useful when investigating a reported startup or import-time
regression.

## How to interpret the output

- **Compare deltas, not absolutes.** Treat the numbers as a relative baseline
  for *your* machine. A meaningful result is a consistent before/after
  difference on the same hardware — not a particular millisecond value.
- **Prefer `median`; watch `p95`.** The median is the stable signal; `p95` and
  `worst` show tail behavior and are noisier.
- **Reduce noise.** Close other heavy processes, keep laptops on AC power (they
  throttle on battery), and run a couple of times — the first `import` and
  agent-build runs are often slower.
- **Don't compare across machines or against CI.** Different CPUs, Python
  builds, and background load make absolute numbers incomparable.

> These measurements are a local diagnostic aid. They are **not** universal
> performance guarantees and should not be quoted as openextract's performance
> characteristics.
