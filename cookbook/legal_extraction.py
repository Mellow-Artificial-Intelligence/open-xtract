from __future__ import annotations

from pydantic import BaseModel

from open_xtract import OpenXtract


class ContractSummary(BaseModel):
    party_a: str
    party_b: str
    effective_date: str
    term_months: int
    governing_law: str
    termination_clause: str


def extract_contract_summary(text: str, ox: OpenXtract | None = None) -> ContractSummary:
    """Extract a structured summary of a legal contract from raw text.

    This uses OpenXtract's text extraction to produce a structured `ContractSummary`.
    Provide your own `OpenXtract` instance if you want to customize the model or settings.
    """
    client = ox or OpenXtract()
    return client.extract_text(text, ContractSummary)


def main() -> None:
    sample_text = (
        "This Master Service Agreement (MSA) is entered into by ACME Corp (\"ACME\") and "
        "BetaCorp (\"BETA\") effective on January 1, 2024. The governing law is Delaware. "
        "The initial term is 12 months. The agreement may be terminated for cause with 30 days notice."
    )
    result = extract_contract_summary(sample_text)
    print(result.model_dump())


__all__ = ["ContractSummary", "extract_contract_summary", "main"]