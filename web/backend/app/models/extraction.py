import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    schema_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("schemas.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    source_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_file_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    schema: Mapped["Schema"] = relationship("Schema", back_populates="extractions")
