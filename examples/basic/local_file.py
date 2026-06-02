"""Extract structured data from a local file path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from _shared import DOCUMENT_PAGE, default_model, require_input
from pydantic import BaseModel

from openextract import extract


class DocumentInfo(BaseModel):
    title: str
    summary: str
    language: str


def main() -> None:
    input_file = require_input(
        sys.argv,
        "Usage: uv run python examples/basic/local_file.py [path-to-file]",
    )
    if input_file == "--fixture":
        input_file = str(DOCUMENT_PAGE)

    result = extract(
        schema=DocumentInfo,
        model=default_model(),
        input_file=input_file,
        instructions=(
            "Read the document and return its title, a two-sentence summary, "
            "and the primary language."
        ),
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()