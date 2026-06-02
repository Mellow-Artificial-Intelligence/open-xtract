"""Shared helpers for runnable examples."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DOCUMENT_PAGE = FIXTURES_DIR / "document_page.png"


def default_model() -> str:
    """Pick a model from OPENEXTRACT_MODEL or whichever provider key is set."""
    if model := os.environ.get("OPENEXTRACT_MODEL"):
        return model
    if os.environ.get("XAI_API_KEY"):
        return "xai:grok-4.3"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter:openai/gpt-4o-mini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai:gpt-4o-mini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic:claude-sonnet-4"
    if os.environ.get("GOOGLE_API_KEY"):
        return "google-gla:gemini-2.5-flash"
    return "xai:grok-4.3"


def require_input(argv: list[str], usage: str) -> str:
    if len(argv) < 2:
        print(usage)
        sys.exit(1)
    return argv[1]


def fixture_path(name: str) -> Path:
    path = FIXTURES_DIR / name
    if not path.is_file():
        print(f"Missing fixture: {path}")
        sys.exit(1)
    return path
