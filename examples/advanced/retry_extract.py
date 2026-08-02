"""Retry extraction on transient model errors with exponential backoff."""

from pydantic import BaseModel

from examples._shared import DOCUMENT_PAGE, openai_model
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
