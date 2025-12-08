"""User management endpoints."""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import delete, select

from app.dependencies import CurrentUser, DbSession
from app.models.provider_credential import ProviderCredential
from app.schemas.user import ProviderCredentialCreate, ProviderCredentialResponse, UserResponse, UserUpdate
from app.utils.security import decrypt_api_key, encrypt_api_key

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserResponse)
async def get_current_user(current_user: CurrentUser) -> UserResponse:
    """Get current user profile."""
    return UserResponse.model_validate(current_user)


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    request: UserUpdate,
    current_user: CurrentUser,
    db: DbSession,
) -> UserResponse:
    """Update current user profile."""
    if request.name is not None:
        current_user.name = request.name
    if request.avatar_url is not None:
        current_user.avatar_url = request.avatar_url

    await db.commit()
    await db.refresh(current_user)

    return UserResponse.model_validate(current_user)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_current_user(current_user: CurrentUser, db: DbSession) -> None:
    """Delete current user account."""
    await db.delete(current_user)
    await db.commit()


# Provider credentials


@router.get("/me/providers", response_model=list[ProviderCredentialResponse])
async def list_provider_credentials(current_user: CurrentUser, db: DbSession) -> list[ProviderCredentialResponse]:
    """List user's configured provider credentials."""
    result = await db.execute(
        select(ProviderCredential)
        .where(ProviderCredential.user_id == current_user.id)
        .order_by(ProviderCredential.provider)
    )
    credentials = result.scalars().all()
    return [ProviderCredentialResponse.model_validate(c) for c in credentials]


@router.post("/me/providers", response_model=ProviderCredentialResponse, status_code=status.HTTP_201_CREATED)
async def add_provider_credential(
    request: ProviderCredentialCreate,
    current_user: CurrentUser,
    db: DbSession,
) -> ProviderCredentialResponse:
    """Add or update a provider API key."""
    # Check if provider already exists for user
    result = await db.execute(
        select(ProviderCredential).where(
            ProviderCredential.user_id == current_user.id,
            ProviderCredential.provider == request.provider,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        # Update existing
        existing.encrypted_api_key = encrypt_api_key(request.api_key)
        existing.is_active = True
        await db.commit()
        await db.refresh(existing)
        return ProviderCredentialResponse.model_validate(existing)

    # Create new
    credential = ProviderCredential(
        user_id=current_user.id,
        provider=request.provider,
        encrypted_api_key=encrypt_api_key(request.api_key),
    )
    db.add(credential)
    await db.commit()
    await db.refresh(credential)

    return ProviderCredentialResponse.model_validate(credential)


@router.delete("/me/providers/{provider}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider_credential(
    provider: str,
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """Delete a provider credential."""
    result = await db.execute(
        delete(ProviderCredential).where(
            ProviderCredential.user_id == current_user.id,
            ProviderCredential.provider == provider,
        )
    )
    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Provider credential not found",
        )


@router.get("/me/providers/available")
async def list_available_providers() -> list[dict]:
    """List all available LLM providers."""
    return [
        {"id": "openai", "name": "OpenAI", "models": ["gpt-4o", "gpt-4o-mini"]},
        {"id": "anthropic", "name": "Anthropic", "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]},
        {"id": "google", "name": "Google", "models": ["gemini-2.0-flash-exp"]},
        {"id": "xai", "name": "xAI", "models": ["grok-beta"]},
        {"id": "openrouter", "name": "OpenRouter", "models": ["qwen/qwen-2.5-72b-instruct"]},
        {"id": "togetherai", "name": "Together AI", "models": []},
        {"id": "groq", "name": "Groq", "models": []},
        {"id": "cerebras", "name": "Cerebras", "models": []},
    ]
