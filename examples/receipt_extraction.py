"""Extract structured receipt fields from an image using openextract."""

import sys
from datetime import date

from pydantic import BaseModel

from openextract import extract


class ReceiptItem(BaseModel):
    name: str
    price: float
    quantity: int = 1


class Receipt(BaseModel):
    merchant: str
    transaction_date: date
    items: list[ReceiptItem]
    subtotal: float
    tax: float
    total: float
    payment_method: str


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/receipt_extraction.py <receipt-image-or-url>")
        sys.exit(1)

    input_file = sys.argv[1]

    receipt = extract(
        schema=Receipt,
        model="xai:grok-4.3",
        input_file=input_file,
        instructions=(
            "Read the receipt image and extract the merchant, transaction date, "
            "each purchased item with its price and quantity, the subtotal, tax, "
            "total, and payment method."
        ),
    )

    print(receipt.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
