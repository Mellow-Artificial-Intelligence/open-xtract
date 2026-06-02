"""Demonstrate catching openextract's typed exceptions (no API call for URL guard)."""

from pydantic import BaseModel

from openextract import ExtractionError, UrlFetchError, extract


class DocumentInfo(BaseModel):
    summary: str


def main() -> None:
    # Private URLs are blocked by default (SSRF protection) before any model call.
    try:
        extract(
            schema=DocumentInfo,
            model="openai:gpt-4o-mini",
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