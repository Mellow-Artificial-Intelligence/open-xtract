"""
Smoke test for openextract against the Gemma paper (arXiv 2403.08295v4).

This is the same document as tests/2403.08295v4.pdf. We use the arXiv URL
because extract() only accepts HTTP/HTTPS URLs — local file paths are
blocked by the SSRF validator.

Run:
    uv run python smoke_test.py

Override the model with an env var if needed:
    OPENEXTRACT_MODEL="google-gla:gemini-2.5-flash" uv run python smoke_test.py
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel

from openextract import extract

load_dotenv()

PAPER_URL = "tests/test.pdf"
DEFAULT_MODEL = "openrouter:google/gemini-2.5-flash-lite-preview-09-2025"


class PaperInfo(BaseModel):
    title: str
    organization: str
    model_family: str
    parameter_sizes: list[str]
    one_sentence_summary: str


def main() -> None:
    model = os.getenv("OPENEXTRACT_MODEL", DEFAULT_MODEL)
    print(f"Extracting from {PAPER_URL} using {model}...\n")

    result = extract(
        schema=PaperInfo,
        model=model,
        input_file_path=PAPER_URL,
        instructions=(
            "Extract the paper's title, the publishing organization, the model "
            "family name, the released parameter sizes (e.g. '2B', '7B'), and a "
            "one-sentence summary of what the paper introduces."
        ),
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
