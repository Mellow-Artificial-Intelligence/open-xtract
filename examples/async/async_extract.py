"""Extract structured data asynchronously with extract_async."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from _shared import DOCUMENT_PAGE, anthropic_model
from pydantic import BaseModel

from openextract import extract_async


class DocumentInfo(BaseModel):
    title: str
    summary: str


async def main() -> None:
    result = await extract_async(
        schema=DocumentInfo,
        model=anthropic_model(),
        input_file=str(DOCUMENT_PAGE),
        instructions="Return a title and a one-sentence summary.",
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
