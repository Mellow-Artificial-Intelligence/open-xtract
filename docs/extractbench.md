---
layout: page
title: ExtractBench
---

# ExtractBench (any model)

[`scripts/extractbench.py`](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/scripts/extractbench.py)
runs [LlamaIndex ExtractBench](https://github.com/run-llama/ExtractBench) through
openextract. Pass any `pydantic-ai` model identifier; the script registers an
openextract pipeline, downloads the dataset if needed, runs inference, and
scores with ExtractBench's official unified value F1.

This is a **quality** benchmark (schema-guided extraction accuracy). It is
separate from [`scripts/bench.py`](benchmarking.md), which only measures local
CPU overhead with the model call mocked out.

## Quick start

From the repository root, with provider credentials in `.env` (the same keys
the CLI and examples use):

```bash
# 6 documents — good for trying a model (cents on a hosted API)
uv run python scripts/extractbench.py --model openai:gpt-5 --test

# One length split
uv run python scripts/extractbench.py --model xai:grok-4.3 --group short

# Full benchmark (370 documents / 4,869 pages — metered API usage)
uv run python scripts/extractbench.py --model anthropic:claude-sonnet-4
```

`--model` follows the same `provider:id` convention as `extract()`, for
example `openai:gpt-5`, `google-gla:gemini-2.5-pro`, `ollama:llama3`, or
`openrouter:anthropic/claude-sonnet-4`. You can also set `OPENEXTRACT_MODEL`
instead of passing `--model`.

The first run creates `.extractbench/venv`, installs ExtractBench from GitHub,
and editable-installs this repo with provider extras. Later runs reuse that
environment. Override the git source with `OPENEXTRACT_EXTRACTBENCH_GIT`.

## What it costs

A full run is 370 documents. Start with `--test`. ExtractBench's own guidance
is that hosted VLMs typically cost on the order of tens of dollars for a full
run; specialized APIs and coding agents cost more. This wrapper records token
usage; pass `--input-price-per-1m` and `--output-price-per-1m` if you want
ExtractBench to compute `cost_usd`. Reported usage and `cost_usd` cover the
successful attempt only; tokens spent on failed attempts that were retried
(`--max-retries`) are not included.

## Output

Results land under `.extractbench/output/<pipeline_name>/` (pipeline names are
`openextract_<slug>` of the model id, or `--pipeline-name`). After a run:

```bash
uv run python scripts/extractbench.py --serve openextract_openai_gpt_5
```

## Other commands

```bash
uv run python scripts/extractbench.py --install          # bootstrap only
uv run python scripts/extractbench.py --download-only --test
uv run python scripts/extractbench.py --status --test
uv run python scripts/extractbench.py --model openai:gpt-5 --skip-inference
```

`--max-concurrent` defaults to 4 (ExtractBench's own default is 20). Raise it
if your provider quota allows. `--max-input-bytes` defaults to 500 MiB so long
scans are not rejected by openextract's 50 MiB library default.

openextract does not currently return per-field bounding boxes, so ExtractBench
**grounding** F1 will be zero. Unified **value** F1 is the score this wrapper
is meant to produce.

## Dataset license

ExtractBench documents come from public records (see the upstream README).
The harness clones ExtractBench into `.extractbench/` (gitignored) and does
not vendor those files in this repository.
