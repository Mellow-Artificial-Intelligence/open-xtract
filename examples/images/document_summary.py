"""Summarize a document page image into structured fields."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from _shared import DOCUMENT_PAGE, default_model
from pydantic import BaseModel

from openextract import extract


class PageSummary(BaseModel):
    headline: str
    bullet_points: list[str]
    language: str


def main() -> None:
    result = extract(
        schema=PageSummary,
        model=default_model(),
        input_file=str(DOCUMENT_PAGE),
        instructions=(
            "Summarize the page: a short headline, 3-5 bullet points of key content, "
            "and the primary language."
        ),
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
