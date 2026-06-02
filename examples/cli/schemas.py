"""Pydantic schemas used by the openextract CLI examples."""

from datetime import date

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    summary: str
    language: str


class Invoice(BaseModel):
    invoice_number: str | None = None
    issue_date: date | None = None
    seller: str | None = None
    total: float | None = None
