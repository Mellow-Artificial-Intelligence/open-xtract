"""Retry extraction on transient model errors with exponential backoff."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from _shared import DOCUMENT_PAGE, openai_model
from pydantic import BaseModel

from openextract import extract


class DocumentInfo(BaseModel):
    summary: str


def main() -> None:
    result = extract(
        schema=DocumentInfo,
        model=openai_model(),
        input_file=str(DOCUMENT_PAGE),
        instructions="Return a one-sentence summary.",
        max_retries=2,
        retry_backoff=0.5,
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
