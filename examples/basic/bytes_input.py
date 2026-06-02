"""Extract structured data from raw bytes with an explicit media type."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from _shared import DOCUMENT_PAGE, default_model
from pydantic import BaseModel

from openextract import extract


class DocumentInfo(BaseModel):
    title: str
    summary: str


def main() -> None:
    data = DOCUMENT_PAGE.read_bytes()

    result = extract(
        schema=DocumentInfo,
        model=default_model(),
        input_file=data,
        media_type="image/png",
        instructions="Return the document title and a one-sentence summary.",
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()