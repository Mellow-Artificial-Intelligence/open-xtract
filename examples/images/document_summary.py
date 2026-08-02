"""Summarize a document page image into structured fields."""

from pydantic import BaseModel

from examples._shared import DOCUMENT_PAGE, xai_model
from openextract import extract


class PageSummary(BaseModel):
    headline: str
    bullet_points: list[str]
    language: str


def main() -> None:
    result = extract(
        schema=PageSummary,
        model=xai_model(),
        input_file=str(DOCUMENT_PAGE),
        instructions=(
            "Summarize the page: a short headline, 3-5 bullet points of key content, "
            "and the primary language."
        ),
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
