"""Demonstrate catching openextract's typed exceptions (no API call for URL guard)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from _shared import openai_model
from pydantic import BaseModel

from openextract import ExtractionError, UrlFetchError, extract


class DocumentInfo(BaseModel):
    summary: str


def main() -> None:
    # Private URLs are blocked by default (SSRF protection) before any model call.
    try:
        extract(
            schema=DocumentInfo,
            model=openai_model(),
            input_file="http://127.0.0.1/internal.pdf",
        )
    except UrlFetchError as exc:
        print(f"UrlFetchError: {exc}")
    except ExtractionError as exc:
        print(f"ExtractionError: {exc}")
    else:
        raise SystemExit("Expected UrlFetchError for private URL")

    print("error_handling example completed successfully")


if __name__ == "__main__":
    main()
