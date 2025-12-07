"""Example: Extract metadata from an academic paper PDF."""

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from open_xtract import OpenXtract


class Author(BaseModel):
    """Author information."""

    name: str
    affiliation: str | None = None


class PaperMetadata(BaseModel):
    """Metadata extracted from an academic paper."""

    title: str = Field(description="The title of the paper")
    authors: list[Author] = Field(description="List of authors")
    abstract: str = Field(description="The paper abstract")
    arxiv_id: str | None = Field(default=None, description="arXiv identifier if present")
    keywords: list[str] = Field(default_factory=list, description="Keywords or topics")


async def main():
    # Path to the test PDF
    pdf_path = Path(__file__).parent.parent / "tests" / "2403.08295v4.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    print(f"Extracting metadata from: {pdf_path.name}")
    print("=" * 60)

    # Initialize extractor
    ox = OpenXtract(
        model="claude-sonnet-4-5",
        system_prompt="Extract academic paper metadata accurately. For authors, include affiliations if visible.",
    )

    # Extract metadata
    result = await ox.extract(pdf_path, PaperMetadata)

    # Display extracted data
    print("\n### Extracted Data ###\n")
    print(f"Title: {result.data.title}")
    print(f"arXiv ID: {result.data.arxiv_id or 'N/A'}")
    print()
    print("Authors:")
    for author in result.data.authors:
        affil = f" ({author.affiliation})" if author.affiliation else ""
        print(f"  - {author.name}{affil}")
    print()
    print("Keywords:", ", ".join(result.data.keywords) if result.data.keywords else "N/A")
    print()
    print("Abstract:")
    abstract = result.data.abstract
    print(abstract[:500] + "..." if len(abstract) > 500 else abstract)

    # Display metadata
    print("\n" + "=" * 60)
    print("### Extraction Metadata ###\n")
    print(f"Model: {result.model or 'N/A'}")
    print(f"Session ID: {result.session_id}")
    print(f"Turns: {result.num_turns}")
    print(f"Duration: {result.duration_ms:,} ms ({result.duration_ms / 1000:.2f}s)")
    print(f"API Duration: {result.duration_api_ms:,} ms ({result.duration_api_ms / 1000:.2f}s)")

    if result.cost_usd is not None:
        print(f"Cost: ${result.cost_usd:.6f}")

    if result.usage:
        print()
        print("Token Usage:")
        print(f"  Input tokens: {result.usage.input_tokens:,}")
        print(f"  Output tokens: {result.usage.output_tokens:,}")
        print(f"  Total tokens: {result.usage.total_tokens:,}")
        if result.usage.cache_read_input_tokens:
            print(f"  Cache read: {result.usage.cache_read_input_tokens:,}")
        if result.usage.cache_creation_input_tokens:
            print(f"  Cache creation: {result.usage.cache_creation_input_tokens:,}")


if __name__ == "__main__":
    asyncio.run(main())
