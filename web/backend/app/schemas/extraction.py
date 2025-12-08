import uuid
from datetime import datetime

from pydantic import BaseModel


class ExtractionCreate(BaseModel):
    schema_id: uuid.UUID
    model: str = "claude-sonnet-4-5"
    file_id: str


class ExtractionResponse(BaseModel):
    id: uuid.UUID
    schema_id: uuid.UUID
    status: str
    model: str
    source_file_name: str | None
    source_file_type: str | None
    source_file_size: int | None
    result: dict | None
    error_message: str | None
    processing_time_ms: int | None
    tokens_used: int | None
    cost_usd: float | None
    created_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class ExtractionListResponse(BaseModel):
    items: list[ExtractionResponse]
    total: int
    page: int
    per_page: int


class FileUploadResponse(BaseModel):
    file_id: str
    file_name: str
    file_type: str
    file_size: int
