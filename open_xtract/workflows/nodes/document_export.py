"""Document export node for converting pages to markdown files."""

import asyncio
from pathlib import Path
from typing import List

import aiofiles

from ...schemas.types import ProcessingState


async def export_documents_node(state: ProcessingState) -> ProcessingState:
    """Export each document page to a markdown file.
    
    Args:
        state: Current processing state
    
    Returns:
        Updated state with markdown file paths
    """
    try:
        if not state.documents:
            state.error = "No documents to export"
            return state
        
        pdf_path = state.pdf_path
        output_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
        output_dir.mkdir(exist_ok=True)
        
        markdown_files: List[Path] = []
        
        # Export each page concurrently
        tasks = []
        for idx, doc in enumerate(state.documents):
            task = _export_page(doc, idx, output_dir, pdf_path.stem)
            tasks.append(task)
        
        markdown_files = await asyncio.gather(*tasks)
        
        state.markdown_files = markdown_files
        return state
        
    except Exception as e:
        state.error = f"Error exporting documents: {str(e)}"
        return state


async def _export_page(doc, page_num: int, output_dir: Path, pdf_name: str) -> Path:
    """Export a single page to markdown."""
    filename = f"{pdf_name}_page_{page_num + 1:03d}.md"
    file_path = output_dir / filename
    
    # Create markdown content
    content = f"# {pdf_name} - Page {page_num + 1}\n\n"
    
    # Add metadata
    content += "## Metadata\n\n"
    for key, value in doc.metadata.items():
        if key not in ["page_content", "original_text"]:
            content += f"- **{key}**: {value}\n"
    content += "\n"
    
    # Add page content
    content += "## Content\n\n"
    content += doc.page_content
    content += "\n"
    
    # If we have original text and it's different, include it
    if "original_text" in doc.metadata and doc.metadata["original_text"] != doc.page_content:
        content += "\n## Original Text (Raw Extraction)\n\n"
        content += doc.metadata["original_text"]
        content += "\n"
    
    # Write file
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        await f.write(content)
    
    return file_path