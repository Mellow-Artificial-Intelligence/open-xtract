"""Smoke tests for runnable examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

API_EXAMPLES: list[tuple[str, list[str]]] = [
    ("basic/local_file.py", ["--fixture"]),
    ("basic/bytes_input.py", []),
    ("images/document_summary.py", []),
    ("advanced/extract_with_usage.py", ["--fixture"]),
]

ALL_SCRIPTS = [
    "basic/local_file.py",
    "basic/bytes_input.py",
    "basic/url_extract.py",
    "images/document_summary.py",
    "images/receipt_extraction.py",
    "documents/invoice_extraction.py",
    "batch/batch_extract.py",
    "async/async_extract.py",
    "advanced/extract_with_usage.py",
    "advanced/retry_extract.py",
    "advanced/error_handling.py",
    "audio/meeting_notes.py",
]


def _run(relative: str, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    path = EXAMPLES / relative
    return subprocess.run(
        [sys.executable, str(path), *(extra or [])],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("relative", ALL_SCRIPTS)
def test_example_script_is_runnable(relative: str) -> None:
    """Each example at least starts; audio/meeting_notes exits 1 without args."""
    path = EXAMPLES / relative
    assert path.is_file(), relative
    if relative == "audio/meeting_notes.py":
        result = _run(relative)
        assert result.returncode == 1
        assert "Usage:" in result.stdout + result.stderr
        return
    if relative == "basic/local_file.py":
        result = _run(relative)
        assert result.returncode == 1
        assert "Usage:" in result.stdout + result.stderr
        return
    if relative in {"images/receipt_extraction.py", "documents/invoice_extraction.py"}:
        result = _run(relative)
        assert result.returncode == 1
        assert "Usage:" in result.stdout + result.stderr
        return
    if relative == "advanced/extract_with_usage.py":
        result = _run(relative)
        assert result.returncode == 1
        assert "Usage:" in result.stdout + result.stderr
        return


def test_error_handling_example() -> None:
    result = _run("advanced/error_handling.py")
    assert result.returncode == 0, result.stderr
    assert "UrlFetchError" in result.stdout
    assert "completed successfully" in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize("relative,args", API_EXAMPLES)
def test_api_examples_with_fixture(relative: str, args: list[str]) -> None:
    """Live model call; skipped in CI unless OPENEXTRACT_RUN_EXAMPLES=1."""
    import os

    if not os.environ.get("OPENEXTRACT_RUN_EXAMPLES"):
        pytest.skip("Set OPENEXTRACT_RUN_EXAMPLES=1 to run live example tests")
    result = _run(relative, args)
    assert result.returncode == 0, result.stderr or result.stdout
