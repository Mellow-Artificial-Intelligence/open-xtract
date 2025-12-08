"""Extraction service that wraps the open-xtract library."""

import asyncio
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.extraction import Extraction
from app.models.schema import Schema
from app.services.schema_builder import build_pydantic_model
from app.services.storage import get_storage_service
from app.services.log_store import log_store


async def run_extraction_background(extraction_id: str) -> None:
    """Run extraction in background with its own database session."""
    from app.database import async_session_maker

    # Small delay to ensure the response is sent first
    await asyncio.sleep(0.1)

    async with async_session_maker() as db:
        try:
            await run_extraction(db, extraction_id)
        except Exception as e:
            await log_store.add_log(extraction_id, f"Background task error: {e}", level="error")
            await log_store.mark_completed(extraction_id)


async def run_extraction(
    db: AsyncSession,
    extraction_id: str,
) -> None:
    """Run an extraction job."""
    from open_xtract import OpenXtract

    await log_store.add_log(extraction_id, "Starting extraction job...")

    result = await db.execute(
        select(Extraction).where(Extraction.id == extraction_id)
    )
    extraction = result.scalar_one_or_none()
    if not extraction:
        await log_store.add_log(extraction_id, "Extraction not found", level="error")
        await log_store.mark_completed(extraction_id)
        return

    extraction.status = "processing"
    await db.commit()
    await log_store.add_log(extraction_id, "Status updated to processing")

    start_time = time.time()

    try:
        await log_store.add_log(extraction_id, "Loading schema...")
        schema_result = await db.execute(
            select(Schema).where(Schema.id == extraction.schema_id)
        )
        schema = schema_result.scalar_one()
        await log_store.add_log(extraction_id, f"Schema loaded: {schema.name}")

        await log_store.add_log(extraction_id, "Building Pydantic model from schema...")
        pydantic_model = build_pydantic_model(schema.fields, schema.name)
        await log_store.add_log(extraction_id, "Pydantic model built successfully")

        await log_store.add_log(extraction_id, "Retrieving source file from storage...")
        storage = get_storage_service()
        file_data = storage.get_file(extraction.source_file_path)
        await log_store.add_log(
            extraction_id,
            f"File retrieved: {extraction.source_file_name} ({len(file_data)} bytes)"
        )

        # Write file to temp location for OpenXtract
        suffix = Path(extraction.source_file_name).suffix if extraction.source_file_name else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        await log_store.add_log(extraction_id, f"File written to temporary location")

        try:
            await log_store.add_log(extraction_id, f"Initializing OpenXtract with model: {extraction.model}")
            ox = OpenXtract(model=extraction.model)

            await log_store.add_log(extraction_id, "Sending document to AI model for extraction...")
            await log_store.add_log(extraction_id, "Waiting for model response...")
            extraction_result = await ox.extract(tmp_path, pydantic_model)
            await log_store.add_log(extraction_id, "Received response from AI model")

            processing_time = int((time.time() - start_time) * 1000)
            extraction.status = "completed"
            extraction.result = extraction_result.data.model_dump()
            extraction.processing_time_ms = processing_time
            extraction.completed_at = datetime.now(timezone.utc)

            if extraction_result.usage:
                extraction.tokens_used = extraction_result.usage.total_tokens
                await log_store.add_log(
                    extraction_id,
                    f"Tokens used: {extraction_result.usage.total_tokens}"
                )
            if extraction_result.cost_usd:
                extraction.cost_usd = extraction_result.cost_usd
                await log_store.add_log(
                    extraction_id,
                    f"Estimated cost: ${extraction_result.cost_usd:.4f}"
                )

            await db.commit()
            await log_store.add_log(
                extraction_id,
                f"Extraction completed successfully in {processing_time}ms",
                level="success"
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            await log_store.add_log(extraction_id, "Cleaned up temporary files")

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        extraction.status = "failed"
        extraction.error_message = str(e)
        extraction.processing_time_ms = processing_time
        extraction.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await log_store.add_log(extraction_id, f"Extraction failed: {str(e)}", level="error")
        await log_store.mark_completed(extraction_id)
        raise
    finally:
        await log_store.mark_completed(extraction_id)


def detect_file_type(file_data: bytes, filename: str) -> str:
    """Detect the type of uploaded file."""
    if file_data[:5] == b"%PDF-":
        return "pdf"

    if file_data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"
    if file_data[:2] == b"\xff\xd8":
        return "image"
    if file_data[:6] in (b"GIF87a", b"GIF89a"):
        return "image"
    if file_data[:4] == b"RIFF" and file_data[8:12] == b"WEBP":
        return "image"

    return "text"
