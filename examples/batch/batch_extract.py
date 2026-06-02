"""Run concurrent extractions over multiple files with extract_many."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from _shared import DOCUMENT_PAGE, default_model
from pydantic import BaseModel

from openextract import extract_many


class DocumentInfo(BaseModel):
    title: str
    summary: str


def main() -> None:
    paths: list[str] = sys.argv[1:] or [str(DOCUMENT_PAGE), str(DOCUMENT_PAGE)]

    results = extract_many(
        schema=DocumentInfo,
        model=default_model(),
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
