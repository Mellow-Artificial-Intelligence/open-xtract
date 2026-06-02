"""Extract structured invoice data from a PDF file using openextract."""

import sys
from datetime import date

from pydantic import BaseModel

from openextract import extract


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total: float


class Invoice(BaseModel):
    invoice_number: str
    issue_date: date
    seller: str
    buyer: str
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/invoice_extraction.py <invoice.pdf>")
        sys.exit(1)

    input_file = sys.argv[1]

    invoice = extract(
        schema=Invoice,
        model="xai:grok-4.3",
        input_file=input_file,
        instructions=(
            "Extract the invoice metadata, parties, every line item, and the "
            "subtotal, tax, and total amounts. Use ISO 8601 for the issue date."
        ),
    )

    print(invoice.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
