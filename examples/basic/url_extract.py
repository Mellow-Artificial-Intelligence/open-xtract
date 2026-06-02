"""Extract structured data from a public HTTPS URL."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from _shared import default_model
from pydantic import BaseModel

from openextract import extract

# Small public PNG used for integration tests (httpbin).
DEFAULT_URL = "https://httpbin.org/image/png"


class ImageInfo(BaseModel):
    description: str
    has_transparency: bool


def main() -> None:
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    if input_file == "--help":
        print(
            "Usage: uv run python examples/basic/url_extract.py [https-url]\n"
            f"Default: {DEFAULT_URL}"
        )
        sys.exit(0)

    result = extract(
        schema=ImageInfo,
        model=default_model(),
        input_file=input_file,
        instructions=(
            "Describe what you see in the image and whether it appears to use transparency."
        ),
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
