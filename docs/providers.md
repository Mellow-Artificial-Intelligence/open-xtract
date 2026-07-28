---
layout: page
title: Provider capability matrix
---

# Provider capability matrix

Status values:

| Status | Meaning |
| ------ | ------- |
| **verified** | Exercised by repo examples or maintainer live smoke runs |
| **expected** | Supported through `pydantic-ai` / the provider SDK, not yet smoke-tested here |
| **unknown** | Possible via upstream, but openextract has not validated it |

Media capabilities inherit from the model and provider SDK. Prefer a
vision/audio-capable model identifier when extracting non-text media.

## Matrix

| Provider | Extra | Credentials | Example model | Text | Image | PDF | Audio | Video | Usage | Notes / evidence |
| -------- | ----- | ----------- | ------------- | ---- | ----- | --- | ----- | ----- | ----- | ---------------- |
| OpenAI | `openai` | `OPENAI_API_KEY` | `openai:gpt-5.5` | verified | verified | expected | expected | unknown | verified | Examples: `basic/local_file.py`, batch, retries. Live smoke: `test_live_openai_image_smoke` (`openai:gpt-5` default). |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `anthropic:claude-opus-4-8` | verified | verified | expected | unknown | unknown | expected | Examples: `basic/bytes_input.py`, `images/receipt_extraction.py`, `documents/invoice_extraction.py`. Live smoke: `test_live_anthropic_image_smoke`. |
| Google (GLA) | `google` | `GEMINI_API_KEY` | `google-gla:gemini-2.5-pro` | expected | expected | expected | expected | unknown | expected | Prefix `google-gla` |
| Google (Vertex) | `google` | GCP ADC / Vertex config | `google-vertex:...` | expected | expected | expected | expected | unknown | expected | Prefix `google-vertex` |
| AWS Bedrock | `bedrock` | AWS creds / `AWS_BEARER_TOKEN` | `bedrock:anthropic.claude-sonnet-4-20250514-v1:0` | expected | expected | expected | unknown | unknown | expected | Region and model access are account-specific |
| xAI | `xai` | `XAI_API_KEY` | `xai:grok-4.3` | verified | verified | expected | verified | unknown | verified | Examples: `basic/url_extract.py`, `images/document_summary.py`, `audio/meeting_notes.py`, usage examples. Live smoke: `test_live_xai_image_smoke`. |
| Cohere | `cohere` | `CO_API_KEY` | `cohere:command-r-plus` | expected | unknown | unknown | unknown | unknown | expected | Multimodal support depends on model |
| Groq | `groq` | `GROQ_API_KEY` | `groq:llama-3.3-70b-versatile` | expected | unknown | unknown | unknown | unknown | expected | Model-dependent media support |
| Hugging Face | `huggingface` | `HF_TOKEN` | `huggingface:meta-llama/Llama-3.3-70B-Instruct` | expected | unknown | unknown | unknown | unknown | expected | Endpoint/model capabilities vary |
| Mistral | `mistral` | `MISTRAL_API_KEY` | `mistral:mistral-large-latest` | expected | expected | expected | unknown | unknown | expected | |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `openrouter:anthropic/claude-sonnet-4` | expected | expected | expected | unknown | unknown | expected | Routes through OpenAI-compatible client |
| Cerebras | `openai` | `CEREBRAS_API_KEY` | `cerebras:llama3.1-70b` | expected | unknown | unknown | unknown | unknown | expected | OpenAI-compatible path |
| Ollama | `openai` | optional `OLLAMA_API_KEY` | `ollama:llama3` | expected | unknown | unknown | unknown | unknown | expected | Local server; uses `NativeOutput` path. Live smoke deferred (needs local daemon). |
| Outlines | install `pydantic-ai-slim[outlines-*]` | backend-specific | `outlines:transformers/...` | expected | unknown | unknown | unknown | unknown | unknown | Local constrained decoding; not an openextract extra |

Example model identifiers match `examples/_shared.py` defaults where applicable
(`openai:gpt-5.5`, `anthropic:claude-opus-4-8`, `xai:grok-4.3`). Live smoke may
use a slightly different OpenAI default (`openai:gpt-5`) — both are illustrative
verified image paths.

Install examples:

```bash
pip install 'openextract[openai]'
pip install 'openextract[anthropic]'
pip install 'openextract[all]'
```

Missing extras raise `ProviderNotInstalledError` with a provider-specific install
hint when the model prefix is known (`_PROVIDER_EXTRAS` in
`src/openextract/_extract.py`).

## Known gaps / follow-ups

- **PDF** cells remain **expected** until a dedicated PDF live smoke exists.
- **Video** cells remain **unknown** / **expected** without a real run.
- **Anthropic usage** stays **expected** (no `extract_with_usage` example against Anthropic yet).
- **Ollama** local smoke needs a running local server; not part of the default ≥3 cloud harness.
- Provider quirks (rate limits, max upload size, vision availability) are owned
  by upstream SDKs and can change without an openextract release.
- See [Live provider smoke tests](live-smoke.md) for the opt-in verification harness.

## Related

- [Troubleshooting](troubleshooting.md)
- [Examples README](https://github.com/Mellow-Artificial-Intelligence/openextract/blob/main/examples/README.md)
