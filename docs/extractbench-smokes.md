---
layout: page
title: ExtractBench smoke log
---

# ExtractBench smoke log

Local 6-document `--test` runs of
[`scripts/extractbench.py`](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/scripts/extractbench.py)
on OpenRouter `z-ai/glm-5.3-flash`. Citations on. The full 370-document
benchmark was not run. PyPI is still `0.12.0`; runner changes are in
CHANGELOG Unreleased.

How to run: [ExtractBench](extractbench.md).

**Gate (not a leaderboard cut):** 6/6 inference finish **and**
successful-only page F1 > 0.37. Not met on the latest run.

When usage was non-zero, `cost_usd` used OpenRouter list prices
$0.075 / $0.25 per 1M input / output tokens.

Scores below are ExtractBench unified **value / page / word** F1.
Successful-only means failed files are omitted from the average, unless
a row says otherwise.

## t211 (this PR)

| | |
| --- | --- |
| When | 2026-09-01, 7:33–8:38 PM CT (~65 min wall) |
| Code | `f01fc2f` on this PR |
| Pipeline | `openextract_glm53flash_t211` |
| Inference | 6/6 OK |
| Successful-only unweighted | value 0.7716, page 0.2374, word 0.1945 (5 docs; Veralto grounded empty) |
| Gate | **not met** (page 0.24 < 0.37) |
| Usage | non-zero; total `cost_usd` $0.515824 |

| Doc | Value | Page | Word | Cites | Wall | cost_usd | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Bianco | 0.7737 | 0.6133 | 0.0000 | 120 (0 bbox) | 1052s | $0.174865 | Compact grayscale PNG, not PDF upload |
| W14 | 0.6431 | 0.0102 | 0.0000 | 5 | 20s | $0.003619 | |
| Goshen | 0.9983 | 0.3510 | 0.5114 | 1411 | 166s | $0.008891 | |
| Veralto | 0.2727 | 0.0000 | — | 249 | 650s | $0.040680 | `output_retries=3`; grounded empty |
| pueblo | 0.9982 | 0.2662 | 0.2735 | 5517 | 1278s | $0.041625 | Leftover merge after 1200s pool budget |
| long | 0.9436 | 0.1838 | 0.1877 | 11768 | 3879s | $0.246144 | Full finish under 4560s pool budget |

## Prior local 6-doc smokes

Same model and cite-on unless a row says otherwise. Successful-only
averages except where noted. Usage `$0` means recorded token counts
were zero.

| SHA / PR | Pipeline era | Finish | Value | Page | Word | Wall | Usage | What failed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `f5a0fa3` (citations PR) | File-upload | 4/6 | 0.91 | 0.37 | 0.00 | ~15.6 min | $0 | 2× OpenRouter 402 |
| `c47d2e8` / #207 | Parse-then-extract | 3/6 | 0.87 | 0.26 | 0.28 | ~46 min | $0 | Bianco + Veralto token-limit |
| `2d83f3e` / #208 | 80k-char windows | 3/6 | 0.86 | 0.13 | 0.17 | ~90 min | $0 | Veralto 1-window token-limit; pueblo/long 3×1800s |
| `53c1711` / #209 | 1-page windows | 1/6 | 0.65 | 0.01 | 0.00 | ~36 min | | 240s wrap; Bianco 400 file-parse |
| `09dcb73` (t210) | Scaled timeouts + scan render | 3/6 | 0.85 | 0.21 | 0.22 | ~92 min | yes | Bianco PNG token-limit; Veralto/long 1800s×3 |
| `e458b2b` (t210b) | File cap ≥ pool; grayscale | 4/6 | 0.68 | 0.08 | 0.11 | ~99 min | yes | Bianco grayscale PNG OK; pueblo 720s pool fail; long 5760s no leftover |
| `0b798b5` / #210 (t210c) | Pool slack +1; no wait on cancel | 3/6 | 0.78 | 0.10 | 0.13 | ~72 min | yes | pueblo 960s no result; long 4320s no leftover; Veralto `output_retries(1)` fail |

t211 is the first run in this series with 6/6 inference finish.
Page F1 is still below the 0.37 gate.
