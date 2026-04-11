"""Core extraction functionality."""

import mimetypes
from pathlib import Path
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, BinaryContent

from .exceptions import ExtractionError, ModelError, SchemaValidationError, UrlFetchError

T = TypeVar("T", bound=BaseModel)


def _get_media_type(file_path: str) -> str:
    """Return the MIME type for a file path (e.g. 'application/pdf')."""
    media_type, _ = mimetypes.guess_type(file_path)
    return media_type or "application/octet-stream"


def _get_media(file_path):
    if file_path.startswith("https://"):
        media = httpx.get(file_path)
        media_bytes = media.content
    else:
        media = Path(file_path)
        media_bytes = media.read_bytes()

    media_type = _get_media_type(file_path=file_path)

    return media_bytes, media_type


def extract(schema: type[T], model: str, input_file: str, instructions: str) -> T:
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
        load_dotenv()
        file_bytes, file_type = _get_media(file_path=input_file)
        agent = Agent(model, instructions=instructions, output_type=schema)
        result = agent.run_sync(
            [
                "Extract the requested information from this document.",
                BinaryContent(data=file_bytes, media_type=file_type),
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
