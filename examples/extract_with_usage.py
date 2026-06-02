"""Extract structured data and print token usage for cost tracking."""

import sys

from pydantic import BaseModel

from openextract import extract_with_usage


class PdfInfo(BaseModel):
    summary: str
    language: str


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/extract_with_usage.py <file-or-url>")
        sys.exit(1)

    input_file = sys.argv[1]

    result, usage = extract_with_usage(
        schema=PdfInfo,
        model="xai:grok-4",
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
