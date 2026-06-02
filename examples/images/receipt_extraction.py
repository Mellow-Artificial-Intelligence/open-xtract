"""Extract structured receipt fields from an image."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
from _shared import DOCUMENT_PAGE, default_model, require_input
from pydantic import BaseModel

from openextract import extract


class ReceiptItem(BaseModel):
    name: str
    price: float
    quantity: int = 1


class Receipt(BaseModel):
    merchant: str
    transaction_date: date | None = None
    items: list[ReceiptItem]
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    payment_method: str | None = None


def main() -> None:
    input_file = require_input(
        sys.argv,
        "Usage: uv run python examples/images/receipt_extraction.py <receipt-image>",
    )
    if input_file == "--fixture":
        input_file = str(DOCUMENT_PAGE)

    receipt = extract(
        schema=Receipt,
        model=default_model(),
        input_file=input_file,
        instructions=(
            "If this is a receipt, extract merchant, date, line items, and totals. "
            "If it is not a receipt, still return the schema with best-effort values "
            "and empty lists where appropriate."
        ),
    )

    print(receipt.model_dump_json(indent=2))


if __name__ == "__main__":
    main()