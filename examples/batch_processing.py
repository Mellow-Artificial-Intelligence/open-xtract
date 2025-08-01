"""Example of batch processing multiple PDFs."""

import asyncio
from pathlib import Path
from typing import List

from pydantic import BaseModel

from open_xtract import PDFProcessor, ExtractionResult


class DocumentMetadata(BaseModel):
    """Generic document metadata schema."""
    document_type: str
    title: str
    date: str
    summary: str = ""
    page_count: int = 0


async def process_batch(pdf_files: List[Path], processor: PDFProcessor) -> List[tuple[Path, ExtractionResult]]:
    """Process multiple PDFs concurrently."""
    tasks = []
    
    for pdf_path in pdf_files:
        task = processor.aprocess_pdf(
            pdf_path=pdf_path,
            schema=DocumentMetadata,
            output_dir=pdf_path.parent / "extracted"
        )
        tasks.append((pdf_path, task))
    
    results = []
    for pdf_path, task in tasks:
        result = await task
        results.append((pdf_path, result))
    
    return results


def main():
    # Find all PDFs in current directory
    pdf_files = list(Path(".").glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in current directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Initialize processor
    processor = PDFProcessor(
        llm_provider="openai",
        model="gpt-4o-mini"
    )
    
    # Process all PDFs
    results = asyncio.run(process_batch(pdf_files, processor))
    
    # Summary
    successful = sum(1 for _, result in results if result.success)
    print(f"\n✓ Successfully processed {successful}/{len(pdf_files)} PDFs")
    
    # Show results
    for pdf_path, result in results:
        print(f"\n{pdf_path.name}:")
        if result.success:
            print(f"  Type: {result.data.document_type}")
            print(f"  Title: {result.data.title}")
            print(f"  Date: {result.data.date}")
            if result.data.summary:
                print(f"  Summary: {result.data.summary[:100]}...")
        else:
            print(f"  ✗ Failed: {result.error}")


if __name__ == "__main__":
    main()