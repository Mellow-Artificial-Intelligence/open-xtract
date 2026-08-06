"""Typing tests for the public API using the ``ty`` type checker.

The library's ``[tool.ty]`` config only type-checks ``src/openextract``. These
tests additionally check a representative consumer module (``tests/typing/
consumer.py``) so the public overloads and contracts are exercised from a
consumer's perspective, matching the issue's "typing tests exercise
representative consumer code" acceptance criterion.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSUMER = ROOT / "tests" / "typing" / "consumer.py"


def test_consumer_code_type_checks() -> None:
    """The representative consumer module must type-check with zero errors.

    The annotated assignments in ``consumer.py`` are the assertions: if the
    batch ``return_exceptions`` overloads or the input/result contracts
    drifted, ``ty`` would report a mismatch here.
    """
    result = subprocess.run(
        [sys.executable, "-m", "ty", "check", str(CONSUMER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ty check failed (exit {result.returncode})\n{result.stdout}\n{result.stderr}"
    )


if __name__ == "__main__":  # pragma: no cover - manual convenience
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
