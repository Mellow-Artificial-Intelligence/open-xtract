#!/usr/bin/env python3
"""Run all examples that work without extra user-provided files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EXAMPLES_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLES_DIR.parent

# Scripts that run with bundled fixtures only (no API key needed for error_handling).
API_EXAMPLES: list[tuple[str, list[str]]] = [
    ("basic/local_file.py", ["--fixture"]),
    ("basic/bytes_input.py", []),
    ("basic/url_extract.py", []),
    ("images/document_summary.py", []),
    ("images/receipt_extraction.py", ["--fixture"]),
    ("documents/invoice_extraction.py", ["--fixture"]),
    ("batch/batch_extract.py", []),
    ("async/async_extract.py", []),
    ("advanced/extract_with_usage.py", ["--fixture"]),
    ("advanced/retry_extract.py", []),
]

NO_API_EXAMPLES: list[str] = [
    "advanced/error_handling.py",
]


def run_script(relative: str, extra_args: list[str] | None = None) -> None:
    path = EXAMPLES_DIR / relative
    cmd = [sys.executable, str(path), *(extra_args or [])]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    print("Running examples that do not require a model API key...")
    for script in NO_API_EXAMPLES:
        run_script(script)

    print(
        "\nRunning examples that call a model "
        "(OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY — or OPENEXTRACT_MODEL to override all)..."
    )
    for script, args in API_EXAMPLES:
        run_script(script, args)

    print("\nAll runnable examples completed successfully.")


if __name__ == "__main__":
    main()
