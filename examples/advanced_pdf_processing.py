"""Advanced PDF processing with different extraction methods."""

from pathlib import Path
from pydantic import BaseModel

from open_xtract import PDFProcessor


class TechnicalDocument(BaseModel):
    """Schema for technical documents with diagrams."""
    title: str
    abstract: str
    figures_count: int = 0
    tables_count: int = 0
    equations: list[str] = []
    references_count: int = 0


def example_with_vision():
    """Example using vision-capable model for better extraction."""
    print("=== Vision-Enhanced Extraction ===")
    
    # This will use PyMuPDF to convert pages to images
    # and use a multimodal model to extract content
    processor = PDFProcessor(
        llm_provider="openai",
        model="gpt-4o"  # Vision-capable model
    )
    
    result = processor.process_pdf(
        pdf_path="technical_paper.pdf",
        schema=TechnicalDocument
    )
    
    if result.success:
        print(f"Title: {result.data.title}")
        print(f"Figures: {result.data.figures_count}")
        print(f"Tables: {result.data.tables_count}")
        if result.data.equations:
            print(f"Equations found: {len(result.data.equations)}")


def example_with_ocr():
    """Example showing how to use OCR for scanned PDFs."""
    print("\n=== OCR-Based Extraction ===")
    
    # For scanned PDFs, vision models work best
    processor = PDFProcessor(
        llm_provider="openai",
        model="gpt-4o-mini"  # Still vision-capable but more cost-effective
    )
    
    # Process a scanned document
    result = processor.process_pdf(
        pdf_path="scanned_invoice.pdf",
        schema={
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "amount": {"type": "number"},
                "vendor": {"type": "string"}
            }
        }
    )
    
    if result.success:
        print(f"Extracted from scanned PDF: {result.data}")


def example_text_only():
    """Example using text-only extraction for simple PDFs."""
    print("\n=== Text-Only Extraction ===")
    
    # For text-based PDFs without complex layouts
    processor = PDFProcessor(
        llm_provider="openai",
        model="gpt-3.5-turbo"  # Non-vision model for cost savings
    )
    
    result = processor.process_pdf(
        pdf_path="simple_report.pdf",
        schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    )
    
    if result.success:
        print(f"Title: {result.data['title']}")
        findings = result.data.get('key_findings', [])
        print(f"Key findings: {len(findings)} items")


def compare_extraction_methods():
    """Compare different extraction methods on the same PDF."""
    print("\n=== Extraction Method Comparison ===")
    
    pdf_path = "sample_document.pdf"
    schema = {"properties": {"content": {"type": "string"}}}
    
    # Method 1: Text-only
    processor_text = PDFProcessor(model="gpt-3.5-turbo")
    result_text = processor_text.process_pdf(pdf_path, schema)
    
    # Method 2: Vision-enhanced
    processor_vision = PDFProcessor(model="gpt-4o")
    result_vision = processor_vision.process_pdf(pdf_path, schema)
    
    if result_text.success and result_vision.success:
        text_len = len(result_text.data.get('content', ''))
        vision_len = len(result_vision.data.get('content', ''))
        
        print(f"Text-only extraction: {text_len} characters")
        print(f"Vision extraction: {vision_len} characters")
        print(f"Difference: {abs(vision_len - text_len)} characters")
        
        # Check metadata
        if result_text.markdown_files:
            with open(result_text.markdown_files[0], 'r') as f:
                content = f.read()
                if "extraction_method: text_only" in content:
                    print("✓ Text-only method confirmed")
        
        if result_vision.markdown_files:
            with open(result_vision.markdown_files[0], 'r') as f:
                content = f.read()
                if "extraction_method: vision_enhanced" in content:
                    print("✓ Vision-enhanced method confirmed")


if __name__ == "__main__":
    # Note: Install vision support with: pip install "open-xtract[vision]"
    
    example_with_vision()
    example_with_ocr()
    example_text_only()
    compare_extraction_methods()