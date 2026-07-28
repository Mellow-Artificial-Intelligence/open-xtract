"""Opt-in live provider smoke tests.

These make real network/model calls. Default pytest/CI runs skip them.

Enable with::

    OPENEXTRACT_LIVE_SMOKE=1 uv run pytest -m integration tests/test_live_smoke.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from openextract import extract

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "fixtures" / "document_page.png"

# Record the identifiers exercised by this harness when opting in.
_LIVE_MODELS = {
    "openai": "openai:gpt-5",
    "anthropic": "anthropic:claude-opus-4-8",
    "xai": "xai:grok-4.3",
}

_SMOKE_INSTRUCTIONS = "Return a one-sentence summary and the primary language."


class _SmokeInfo(BaseModel):
    summary: str
    language: str


def _live_enabled() -> bool:
    return os.environ.get("OPENEXTRACT_LIVE_SMOKE", "").lower() in ("1", "true", "yes")


def _require_live(credential_env: str) -> None:
    if not _live_enabled():
        pytest.skip("Set OPENEXTRACT_LIVE_SMOKE=1 to run live provider smoke tests")
    if not os.environ.get(credential_env):
        pytest.skip(f"{credential_env} is required for this live smoke test")
    if not FIXTURE.is_file():
        pytest.skip(f"missing fixture: {FIXTURE}")


def _run_image_smoke(*, model: str) -> None:
    result = extract(
        schema=_SmokeInfo,
        model=model,
        input_file=str(FIXTURE),
        instructions=_SMOKE_INSTRUCTIONS,
    )
    assert isinstance(result.summary, str) and result.summary.strip()
    assert isinstance(result.language, str) and result.language.strip()


@pytest.mark.integration
def test_live_openai_image_smoke() -> None:
    """OpenAI-compatible cloud image path; skipped unless explicitly enabled."""
    _require_live("OPENAI_API_KEY")
    model = os.environ.get("OPENEXTRACT_LIVE_MODEL_OPENAI") or os.environ.get(
        "OPENEXTRACT_LIVE_MODEL", _LIVE_MODELS["openai"]
    )
    _run_image_smoke(model=model)


@pytest.mark.integration
def test_live_anthropic_image_smoke() -> None:
    """Non-OpenAI cloud image path (Anthropic); skipped unless explicitly enabled."""
    _require_live("ANTHROPIC_API_KEY")
    model = os.environ.get("OPENEXTRACT_LIVE_MODEL_ANTHROPIC", _LIVE_MODELS["anthropic"])
    _run_image_smoke(model=model)


@pytest.mark.integration
def test_live_xai_image_smoke() -> None:
    """Third representative cloud path (xAI); skipped unless explicitly enabled.

    A local Ollama path remains deferred: it needs a running local server and is
    covered separately when maintainers opt into OpenAI-compatible local smoke.
    """
    _require_live("XAI_API_KEY")
    model = os.environ.get("OPENEXTRACT_LIVE_MODEL_XAI", _LIVE_MODELS["xai"])
    _run_image_smoke(model=model)
