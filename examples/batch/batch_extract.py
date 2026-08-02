"""Run concurrent extractions over multiple files with extract_many."""

import sys

from pydantic import BaseModel

from examples._shared import DOCUMENT_PAGE, openai_model
from openextract import extract_many


class DocumentInfo(BaseModel):
    title: str
    summary: str


def main() -> None:
    paths: list[str] = sys.argv[1:] or [str(DOCUMENT_PAGE), str(DOCUMENT_PAGE)]

    results = extract_many(
        schema=DocumentInfo,
        model=openai_model(),
        input_files=paths,
        max_concurrency=2,
        instructions="Return a short title and one-sentence summary.",
    )

    for path, doc in zip(paths, results, strict=True):
        print(f"=== {path} ===")
        print(doc.model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    main()
