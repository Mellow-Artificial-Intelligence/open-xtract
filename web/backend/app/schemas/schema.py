import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FieldDefinition(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: Literal["string", "integer", "float", "boolean", "array", "object"]
    description: str | None = None
    required: bool = True
    default: str | int | float | bool | None = None
    items: "FieldDefinition | None" = None
    fields: list["FieldDefinition"] | None = None


class SchemaCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    fields: list[FieldDefinition]


class SchemaUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    fields: list[FieldDefinition] | None = None


class SchemaResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    version: int
    fields: dict
    created_at: datetime
    updated_at: datetime | None

    class Config:
        from_attributes = True


class SchemaListResponse(BaseModel):
    items: list[SchemaResponse]
    total: int
    page: int
    per_page: int
