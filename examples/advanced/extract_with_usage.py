"""Extract structured data and print token usage for cost tracking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from _shared import DOCUMENT_PAGE, require_input, xai_model
from pydantic import BaseModel

from openextract import extract_with_usage


class DocumentInfo(BaseModel):
    summary: str
    language: str


def main() -> None:
    input_file = require_input(
        sys.argv,
        "Usage: uv run python examples/advanced/extract_with_usage.py [path-or-url]",
    )
    if input_file == "--fixture":
        input_file = str(DOCUMENT_PAGE)

    result, usage = extract_with_usage(
        schema=DocumentInfo,
        model=xai_model(),
        input_file=input_file,
        instructions="Return a two-sentence summary and the document's primary language.",
    )

    print(result.model_dump_json(indent=2))
    print(
        f"\ntokens: {usage.input_tokens} in / {usage.output_tokens} out / "
        f"{usage.total_tokens} total"
    )


if __name__ == "__main__":
    main()
