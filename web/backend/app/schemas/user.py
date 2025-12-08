import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    name: str | None
    avatar_url: str | None
    auth_provider: str
    email_verified_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    name: str | None = None
    avatar_url: str | None = None


class ProviderCredentialCreate(BaseModel):
    provider: str
    api_key: str


class ProviderCredentialResponse(BaseModel):
    id: uuid.UUID
    provider: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
