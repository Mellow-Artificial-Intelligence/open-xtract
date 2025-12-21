"""Core extraction functionality."""

import os
from typing import TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, AudioUrl, DocumentUrl, ImageUrl, VideoUrl

from .exceptions import ExtractionError, ModelError, SchemaValidationError, UrlFetchError

T = TypeVar("T", bound=BaseModel)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv"}
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".html", ".csv", ".xls", ".xlsx"}


def _get_media_url(url: str):
    """Determine the appropriate media URL type based on file extension."""
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        return ImageUrl(url=url)
    elif ext in AUDIO_EXTENSIONS:
        return AudioUrl(url=url)
    elif ext in VIDEO_EXTENSIONS:
        return VideoUrl(url=url)
    else:
        return DocumentUrl(url=url)


def extract(schema: type[T], model: str, url: str, instructions: str) -> T:
    """
    Extract structured data from a URL using an LLM.

    Args:
        schema: A Pydantic model class defining the expected output structure.
        model: The model identifier (e.g., 'google-gla:gemini-3-flash-preview').
        url: The URL of the document, image, audio, or video to extract from.
        instructions: Instructions for the LLM on what to extract.

    Returns:
        An instance of the schema populated with extracted data.

    Raises:
        UrlFetchError: If the URL cannot be fetched.
        SchemaValidationError: If the model output doesn't match the schema.
        ModelError: If there's an error communicating with the model API.
        ExtractionError: For other extraction failures.
    """
    try:
        agent = Agent(model, instructions=instructions, output_type=schema)
        media_url = _get_media_url(url)
        result = agent.run_sync(
            [
                "Extract the requested information from this document.",
                media_url,
            ]
        )
        return result.output
    except httpx.HTTPStatusError as e:
        raise UrlFetchError(f"Failed to fetch URL: {e.response.status_code}") from e
    except httpx.RequestError as e:
        raise UrlFetchError(f"Failed to fetch URL: {e}") from e
    except ValidationError as e:
        raise SchemaValidationError(f"Model output did not match schema: {e}") from e
    except Exception as e:
        if "api" in str(type(e).__module__).lower() or "model" in str(e).lower():
            raise ModelError(f"Model API error: {e}") from e
        raise ExtractionError(f"Extraction failed: {e}") from e
