---
layout: page
title: Design — API reference for 1.0
---

# Design: API reference surface for 1.0

**Status:** accepted decision  
**Related issue:** [#146](https://github.com/Mellow-Artificial-Intelligence/openextract/issues/146)

## Decision

**README-only API reference is sufficient for 1.0.**

Do **not** add a generated `pdoc` / Sphinx / mkdocs API tree before 1.0 unless a
follow-up issue explicitly reopens this after freeze.

## Why

1. `README.md` already documents every public function, `Usage`, and exception
   category that appears in `openextract.__all__`.
2. The **Public API stability** table in the README is the 1.0 contract map and
   must stay hand-authored.
3. CLI contracts live in `docs/cli.md` and are not generated from Python
   docstrings.
4. A generated docs tree would need CI freshness gates and duplication of the
   stability / compatibility text without improving the 1.0 freeze.

## What stays hand-authored

| Surface | Location |
| ------- | -------- |
| Function args / returns / errors | `README.md` → API reference |
| Stability (Stable vs Provisional) | `README.md` → Public API stability |
| Compatibility / deprecation | `README.md` → Compatibility and deprecation policy |
| CLI exit codes / streams | `docs/cli.md` |
| Provider matrix | `docs/providers.md` |

## Sync rules (pre-1.0 and after)

1. Any public signature change in `src/openextract/_extract.py` or
   `src/openextract/exceptions.py` **must** update the README API reference and
   stability table in the same PR.
2. `__all__` changes require a README stability-table row add/remove.
3. CLI flag or exit-code changes require `docs/cli.md` + `tests/test_cli.py` in
   the same PR.
4. Optional later: a CI check that `__all__` symbols appear in the README
   stability table (string presence), without generating HTML.

## Gaps to fix before freeze (tracked elsewhere)

- Document `max_input_bytes` / `InputTooLargeError` / `OPENEXTRACT_MAX_INPUT_BYTES`
  once implemented (#143).
- Keep example model IDs aligned with `examples/_shared.py` and the provider
  matrix (#145).

## Deferred (post-1.0 optional)

- Generated API pages from docstrings, only if maintainers want deeper
  navigation than the README.
- If revisited, prefer a single `docs/api.md` generated in CI over a multi-page
  Sphinx tree.
