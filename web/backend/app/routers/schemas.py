"""Schema management endpoints."""

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from app.dependencies import DbSession
from app.models.schema import Schema
from app.schemas.schema import SchemaCreate, SchemaListResponse, SchemaResponse, SchemaUpdate
from app.services.schema_builder import schema_to_pydantic_code, validate_schema_definition

router = APIRouter(prefix="/schemas", tags=["schemas"])


@router.get("", response_model=SchemaListResponse)
async def list_schemas(
    db: DbSession,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
) -> SchemaListResponse:
    """List all schemas with pagination."""
    offset = (page - 1) * per_page

    count_result = await db.execute(select(func.count()).select_from(Schema))
    total = count_result.scalar() or 0

    result = await db.execute(
        select(Schema)
        .order_by(Schema.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    schemas = result.scalars().all()

    return SchemaListResponse(
        items=[SchemaResponse.model_validate(s) for s in schemas],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.post("", response_model=SchemaResponse, status_code=status.HTTP_201_CREATED)
async def create_schema(
    request: SchemaCreate,
    db: DbSession,
) -> SchemaResponse:
    """Create a new extraction schema."""
    fields_dict = {"fields": [f.model_dump() for f in request.fields]}
    errors = validate_schema_definition(fields_dict)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": errors},
        )

    schema = Schema(
        name=request.name,
        description=request.description,
        fields=fields_dict,
    )
    db.add(schema)
    await db.commit()
    await db.refresh(schema)

    return SchemaResponse.model_validate(schema)


@router.post("/validate")
async def validate_schema(request: SchemaCreate) -> dict:
    """Validate a schema definition without saving."""
    fields_dict = {"fields": [f.model_dump() for f in request.fields]}
    errors = validate_schema_definition(fields_dict)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


@router.post("/preview")
async def preview_schema(request: SchemaCreate) -> dict:
    """Preview the generated Pydantic model code."""
    fields_dict = {"fields": [f.model_dump() for f in request.fields]}
    errors = validate_schema_definition(fields_dict)

    if errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": errors},
        )

    code = schema_to_pydantic_code(fields_dict, request.name.replace(" ", ""))

    return {"code": code}


@router.get("/{schema_id}", response_model=SchemaResponse)
async def get_schema(
    schema_id: uuid.UUID,
    db: DbSession,
) -> SchemaResponse:
    """Get a schema by ID."""
    result = await db.execute(select(Schema).where(Schema.id == schema_id))
    schema = result.scalar_one_or_none()

    if not schema:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schema not found",
        )

    return SchemaResponse.model_validate(schema)


@router.put("/{schema_id}", response_model=SchemaResponse)
async def update_schema(
    schema_id: uuid.UUID,
    request: SchemaUpdate,
    db: DbSession,
) -> SchemaResponse:
    """Update a schema."""
    result = await db.execute(select(Schema).where(Schema.id == schema_id))
    schema = result.scalar_one_or_none()

    if not schema:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schema not found",
        )

    if request.name is not None:
        schema.name = request.name
    if request.description is not None:
        schema.description = request.description
    if request.fields is not None:
        fields_dict = {"fields": [f.model_dump() for f in request.fields]}
        errors = validate_schema_definition(fields_dict)
        if errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"errors": errors},
            )
        schema.fields = fields_dict
        schema.version += 1

    await db.commit()
    await db.refresh(schema)

    return SchemaResponse.model_validate(schema)


@router.delete("/{schema_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schema(
    schema_id: uuid.UUID,
    db: DbSession,
) -> None:
    """Delete a schema."""
    result = await db.execute(select(Schema).where(Schema.id == schema_id))
    schema = result.scalar_one_or_none()

    if not schema:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schema not found",
        )

    await db.delete(schema)
    await db.commit()


@router.post("/{schema_id}/duplicate", response_model=SchemaResponse, status_code=status.HTTP_201_CREATED)
async def duplicate_schema(
    schema_id: uuid.UUID,
    db: DbSession,
) -> SchemaResponse:
    """Duplicate a schema."""
    result = await db.execute(select(Schema).where(Schema.id == schema_id))
    original = result.scalar_one_or_none()

    if not original:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schema not found",
        )

    new_schema = Schema(
        name=f"{original.name} (Copy)",
        description=original.description,
        fields=original.fields,
    )
    db.add(new_schema)
    await db.commit()
    await db.refresh(new_schema)

    return SchemaResponse.model_validate(new_schema)
