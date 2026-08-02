"""Extract structured data asynchronously with extract_async."""

import asyncio

from pydantic import BaseModel

from examples._shared import DOCUMENT_PAGE, anthropic_model
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
