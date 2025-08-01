"""Basic example of extracting invoice data from a PDF."""

from datetime import datetime
from pydantic import BaseModel

from open_xtract import PDFProcessor


class InvoiceData(BaseModel):
    """Schema for invoice data extraction."""
    invoice_number: str
    date: str
    total_amount: float
    vendor: str
    customer_name: str = ""
    items_count: int = 0


def main():
    # Initialize processor
    processor = PDFProcessor(
        llm_provider="openai",  # or any OpenAI-compatible provider
        model="gpt-4o-mini"     # or gpt-4o for better accuracy
    )
    
    # Process PDF
    result = processor.process_pdf(
        pdf_path="invoice.pdf",
        schema=InvoiceData
    )
    
    if result.success:
        print("✓ Successfully extracted invoice data!")
        print(f"\nInvoice Number: {result.data.invoice_number}")
        print(f"Date: {result.data.date}")
        print(f"Vendor: {result.data.vendor}")
        print(f"Total Amount: ${result.data.total_amount:.2f}")
        
        if result.data.customer_name:
            print(f"Customer: {result.data.customer_name}")
        
        if result.data.items_count:
            print(f"Number of items: {result.data.items_count}")
        
        print(f"\nMarkdown files saved: {len(result.markdown_files)}")
    else:
        print(f"✗ Extraction failed: {result.error}")


if __name__ == "__main__":
    main()