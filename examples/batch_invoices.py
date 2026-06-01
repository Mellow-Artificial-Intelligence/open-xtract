"""Extract structured invoice data from many PDFs concurrently."""

import sys
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from openextract import extract_many


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
        print("Usage: uv run python examples/batch_invoices.py <dir-or-pdf> [...]")
        sys.exit(1)

    paths: list[str] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(str(f) for f in sorted(p.glob("*.pdf")))
        else:
            paths.append(str(p))

    if not paths:
        print("No PDF files found.")
        sys.exit(1)

    results = extract_many(
        schema=Invoice,
        model="openai:gpt-5",
        input_files=paths,
        max_concurrency=5,
        instructions=(
            "Extract invoice metadata, parties, line items, and totals. "
            "Use ISO 8601 for issue_date."
        ),
    )

    for path, invoice in zip(paths, results, strict=True):
        print(f"=== {path} ===")
        print(invoice.model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    main()
