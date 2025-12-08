"""FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

DbSession = Annotated[AsyncSession, Depends(get_db)]
