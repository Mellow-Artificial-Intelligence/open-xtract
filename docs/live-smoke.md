---
layout: page
title: Live provider smoke tests
---

# Live provider smoke tests

Optional harness for maintainers to verify real provider integration paths.
Default `pytest` / CI runs never make network or model calls.

## Opt-in

```bash
# representative live smoke (requires credentials for the selected model)
OPENEXTRACT_LIVE_SMOKE=1 uv run pytest -m integration tests/test_live_smoke.py -v

# existing example-based live checks
OPENEXTRACT_RUN_EXAMPLES=1 uv run pytest -m integration tests/test_examples.py -v
```

Without these env vars, integration tests skip.

## What is covered

| Test | Model (default) | Media | Credential | Override |
| ---- | --------------- | ----- | ---------- | -------- |
| `test_live_openai_image_smoke` | `openai:gpt-5` | bundled PNG fixture | `OPENAI_API_KEY` | `OPENEXTRACT_LIVE_MODEL_OPENAI` or `OPENEXTRACT_LIVE_MODEL` |
| `test_live_anthropic_image_smoke` | `anthropic:claude-opus-4-8` | bundled PNG fixture | `ANTHROPIC_API_KEY` | `OPENEXTRACT_LIVE_MODEL_ANTHROPIC` |
| `test_live_xai_image_smoke` | `xai:grok-4.3` | bundled PNG fixture | `XAI_API_KEY` | `OPENEXTRACT_LIVE_MODEL_XAI` |

A fourth local/Ollama path is deferred: it needs a running local server and is
not required for the ≥3 representative-provider gate.

When cutting a release that gates on provider compatibility, record the verified
model IDs from this harness in the release notes / changelog.

## Design rules

- Marked `@pytest.mark.integration`.
- Explicit env opt-in only.
- Fixtures are small and non-sensitive (`examples/fixtures/`).
- Not enabled in default CI.
- Promote matrix cells in [providers.md](providers.md) from expected → verified
  when a path is stable.

## Related

- [Provider capability matrix](providers.md)
- [Examples](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/examples/README.md)
