"""Extraction endpoints."""

import asyncio
import json
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from app.database import async_session_maker
from app.dependencies import DbSession
from app.models.extraction import Extraction
from app.models.schema import Schema
from app.schemas.extraction import (
    ExtractionCreate,
    ExtractionListResponse,
    ExtractionResponse,
    FileUploadResponse,
)
from app.services.extraction import detect_file_type, run_extraction_background
from app.services.storage import get_storage_service
from app.services.log_store import log_store

router = APIRouter(prefix="/extractions", tags=["extractions"])


@router.get("", response_model=ExtractionListResponse)
async def list_extractions(
    db: DbSession,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    schema_id: uuid.UUID | None = None,
    status_filter: str | None = Query(None, alias="status"),
) -> ExtractionListResponse:
    """List extractions with pagination and filtering."""
    offset = (page - 1) * per_page

    query = select(Extraction)
    count_query = select(func.count()).select_from(Extraction)

    if schema_id:
        query = query.where(Extraction.schema_id == schema_id)
        count_query = count_query.where(Extraction.schema_id == schema_id)

    if status_filter:
        query = query.where(Extraction.status == status_filter)
        count_query = count_query.where(Extraction.status == status_filter)

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    result = await db.execute(
        query.order_by(Extraction.created_at.desc()).offset(offset).limit(per_page)
    )
    extractions = result.scalars().all()

    return ExtractionListResponse(
        items=[ExtractionResponse.model_validate(e) for e in extractions],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
) -> FileUploadResponse:
    """Upload a file for extraction."""
    max_size = 50 * 1024 * 1024
    content = await file.read()

    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 50MB.",
        )

    file_type = detect_file_type(content, file.filename or "file")

    content_type_map = {
        "pdf": "application/pdf",
        "image": "image/png",
        "text": "text/plain",
    }
    content_type = file.content_type or content_type_map.get(file_type, "application/octet-stream")

    storage = get_storage_service()
    file_path = storage.upload_file(
        file_data=content,
        file_name=file.filename or "uploaded_file",
        content_type=content_type,
    )

    return FileUploadResponse(
        file_id=file_path,
        file_name=file.filename or "uploaded_file",
        file_type=file_type,
        file_size=len(content),
    )


@router.post("", response_model=ExtractionResponse, status_code=status.HTTP_201_CREATED)
async def create_extraction(
    request: ExtractionCreate,
    db: DbSession,
) -> ExtractionResponse:
    """Create a new extraction job."""
    result = await db.execute(select(Schema).where(Schema.id == request.schema_id))
    schema = result.scalar_one_or_none()

    if not schema:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schema not found",
        )

    storage = get_storage_service()
    if not storage.file_exists(request.file_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File not found. Please upload the file first.",
        )

    parts = request.file_id.split("/")
    file_name = parts[-1] if parts else "file"

    file_data = storage.get_file(request.file_id)
    file_type = detect_file_type(file_data, file_name)

    extraction = Extraction(
        schema_id=request.schema_id,
        model=request.model,
        status="pending",
        source_file_path=request.file_id,
        source_file_name=file_name,
        source_file_type=file_type,
        source_file_size=len(file_data),
    )
    db.add(extraction)
    await db.commit()
    await db.refresh(extraction)

    # Run extraction in background so logs can stream in real-time
    asyncio.create_task(run_extraction_background(str(extraction.id)))

    return ExtractionResponse.model_validate(extraction)


@router.get("/stats")
async def get_extraction_stats(db: DbSession) -> dict:
    """Get extraction statistics."""
    total_result = await db.execute(select(func.count()).select_from(Extraction))
    total = total_result.scalar() or 0

    status_result = await db.execute(
        select(Extraction.status, func.count())
        .group_by(Extraction.status)
    )
    by_status = {row[0]: row[1] for row in status_result.all()}

    avg_time_result = await db.execute(
        select(func.avg(Extraction.processing_time_ms))
        .where(Extraction.status == "completed")
    )
    avg_time = avg_time_result.scalar()

    return {
        "total": total,
        "by_status": by_status,
        "avg_processing_time_ms": int(avg_time) if avg_time else None,
    }


@router.get("/{extraction_id}", response_model=ExtractionResponse)
async def get_extraction(
    extraction_id: uuid.UUID,
    db: DbSession,
) -> ExtractionResponse:
    """Get an extraction by ID."""
    result = await db.execute(select(Extraction).where(Extraction.id == extraction_id))
    extraction = result.scalar_one_or_none()

    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction not found",
        )

    return ExtractionResponse.model_validate(extraction)


@router.delete("/{extraction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_extraction(
    extraction_id: uuid.UUID,
    db: DbSession,
) -> None:
    """Delete an extraction."""
    result = await db.execute(select(Extraction).where(Extraction.id == extraction_id))
    extraction = result.scalar_one_or_none()

    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction not found",
        )

    if extraction.source_file_path:
        try:
            storage = get_storage_service()
            storage.delete_file(extraction.source_file_path)
        except Exception:
            pass

    await db.delete(extraction)
    await db.commit()


@router.post("/{extraction_id}/retry", response_model=ExtractionResponse)
async def retry_extraction(
    extraction_id: uuid.UUID,
    db: DbSession,
) -> ExtractionResponse:
    """Retry a failed extraction."""
    result = await db.execute(select(Extraction).where(Extraction.id == extraction_id))
    extraction = result.scalar_one_or_none()

    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction not found",
        )

    if extraction.status != "failed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only retry failed extractions",
        )

    extraction.status = "pending"
    extraction.error_message = None
    extraction.result = None
    extraction.processing_time_ms = None
    extraction.completed_at = None
    await db.commit()
    await db.refresh(extraction)

    # Run extraction in background so logs can stream in real-time
    asyncio.create_task(run_extraction_background(str(extraction.id)))

    return ExtractionResponse.model_validate(extraction)


@router.get("/{extraction_id}/file")
async def get_extraction_file(
    extraction_id: uuid.UUID,
    db: DbSession,
) -> dict:
    """Get presigned URL for downloading the source file."""
    result = await db.execute(select(Extraction).where(Extraction.id == extraction_id))
    extraction = result.scalar_one_or_none()

    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Extraction not found",
        )

    if not extraction.source_file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source file not found",
        )

    storage = get_storage_service()
    url = storage.get_presigned_url(extraction.source_file_path)

    return {"url": url, "filename": extraction.source_file_name}


@router.get("/{extraction_id}/logs")
async def get_extraction_logs(extraction_id: uuid.UUID) -> list[dict]:
    """Get all logs for an extraction."""
    return await log_store.get_logs(str(extraction_id))


@router.get("/{extraction_id}/logs/stream")
async def stream_extraction_logs(extraction_id: uuid.UUID):
    """Stream logs for an extraction using Server-Sent Events."""

    async def event_generator():
        async for entry in log_store.subscribe(str(extraction_id)):
            if entry is None:
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            yield f"data: {json.dumps({'type': 'log', 'entry': entry.to_dict()})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
