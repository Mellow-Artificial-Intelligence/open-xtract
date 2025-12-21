import os
from typing import TypeVar
from urllib.parse import urlparse

import httpx
import logfire
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, AudioUrl, DocumentUrl, ImageUrl, VideoUrl

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

T = TypeVar("T", bound=BaseModel)


class ExtractionError(Exception):
    """Base exception for extraction errors."""

    pass


class ModelError(ExtractionError):
    """Error communicating with the model API."""

    pass


class SchemaValidationError(ExtractionError):
    """Model output did not match the expected schema."""

    pass


class UrlFetchError(ExtractionError):
    """Error fetching the URL content."""

    pass


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv"}
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".html", ".csv", ".xls", ".xlsx"}


def _get_media_url(url: str):
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


class PdfInfo(BaseModel):
    summary: str
    language: str


if __name__ == "__main__":
    result = extract(
        schema=PdfInfo,
        model="google-gla:gemini-3-flash-preview",
        url="https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
        instructions="return a 2 sentence summary and the primary language of the document",
    )
    print(result)
