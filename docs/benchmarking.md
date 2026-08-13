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

For the startup and first-error measurements only:

```bash
uv run python scripts/bench.py --startup-only
```

To compare a base install with the full provider development environment
without changing the repository's normal virtual environment:

```bash
uv run --isolated --no-dev --locked python scripts/bench.py --startup-only
uv run python scripts/bench.py --startup-only
```

## What it measures

Most rows report `median`, `p95`, and `best` over `n` iterations (after a
warmup), so both typical and tail cost are visible. Startup rows instead report
`median`, `best`, and `worst` over five fresh subprocesses:

- **`[environment]`** — revision, Python, platform, and installed provider SDK
  versions needed to interpret or reproduce a result.
- **`[import]`** — cold `import openextract` latency and maximum resident set
  size (RSS).
- **`[model error]`** — first provider-neutral error classification latency,
  RSS growth, and any provider modules imported by classification.
- **`[_get_media]`** — resolving an `input_file` to `(bytes, media_type)` for a
  small text file, a ~50 KB PDF, a ~5 MB PDF, and raw `bytes` input.
- **`[_get_media_type]`** — `mimetypes.guess_type` on a filename.
- **`[_build_agent]`** — constructing a `pydantic_ai.Agent` for a non-Ollama
  and an Ollama model.
- **`[extract]` / `[extract_many]`** — the full `extract()` / `extract_many()`
  path with the `Agent` **mocked out**: all per-call local cost (media read +
  agent build + dispatch) *except* the model call.

## What it intentionally does NOT measure

- **Model / network / inference latency.** The LLM round-trip is mocked. In
  real usage it dominates total time by orders of magnitude; this tool
  deliberately excludes it so the local CPU costs are visible at all.
- **Provider SDK network calls**, rate limits, retries, or token costs.
- **Cross-machine or absolute performance.** Numbers reflect the specific
  machine, Python build, and system load at the moment of the run.

## Issue #165 before/after record

Measurements below were captured on 2026-08-03 with CPython 3.12.9 on Apple
silicon/macOS 26.5.1. They are medians of fresh subprocess samples in the full
provider development environment. The base-install result is recorded
separately because environment composition materially affects RSS.

| Profile / path | Before | After | Change |
| --- | ---: | ---: | ---: |
| Full-provider cold import | 262.97 ms / 71.91 MiB | 61.21 ms / 41.94 MiB | -77% latency / -42% RSS |
| First model-error classification | 672.62 ms / +102.59 MiB | 44.58 us / +0.00 MiB | >99.99% latency reduction |
| Base-install cold import | not recorded | 68.16 ms / 44.77 MiB | baseline established |

Before the change, classifying one Pydantic AI `ModelAPIError` imported OpenAI,
Anthropic, Google GenAI, Botocore, Cohere, Hugging Face, Groq, Mistral, and gRPC
exception modules. After the change, the benchmark reports `none`: the
classifier matches exact module/class signatures already present in the
exception's MRO. Public error types and provider mappings are unchanged.

These results establish investigation budgets for comparable Apple-silicon
development runs:

- cold-import median at or below 100 ms and median max RSS at or below 55 MiB,
- first-error classification median below 1 ms, RSS growth below 1 MiB, and
  zero newly imported provider modules.

The time and RSS values are diagnostic budgets, not CI assertions or public
performance guarantees. The zero-provider-import invariant is deterministic
and is enforced by the test suite.

## When maintainers should run it

Run it **before and after** a change that touches the local hot path, and
compare the two runs on the **same machine**:

- changes to `import openextract` or module-level import work,
- provider exception mapping or optional-provider loading,
- `_get_media` / media handling,
- `_build_agent` / agent construction,
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

## Extraction quality (ExtractBench)

For schema-guided extraction accuracy against LlamaIndex ExtractBench, use
[`scripts/extractbench.py`](extractbench.md) with any `pydantic-ai` model
identifier. That tool calls live models and is not a substitute for this
local overhead benchmark.
