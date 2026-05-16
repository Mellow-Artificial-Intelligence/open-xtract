"""Core extraction functionality."""

import mimetypes
from pathlib import Path
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.output import NativeOutput

from .exceptions import ExtractionError, ModelError, SchemaValidationError, UrlFetchError

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MEDIA_TYPE = "application/octet-stream"
_URL_PREFIXES = ("http://", "https://")
_URL_FETCH_TIMEOUT = 30.0


def _collect_model_error_types() -> tuple[type[BaseException], ...]:
    """Collect known provider/model error base classes that are importable.

    Each import is guarded so a missing optional provider does not break the
    package. The returned tuple is suitable for use with ``isinstance``.
    """
    error_types: list[type[BaseException]] = []

    try:
        from pydantic_ai.exceptions import ModelAPIError

        error_types.append(ModelAPIError)
    except ImportError:  # pragma: no cover - pydantic-ai is a hard dependency
        pass

    try:
        from openai import APIError as OpenAIAPIError

        error_types.append(OpenAIAPIError)
    except ImportError:  # pragma: no cover - openai extra is installed
        pass

    try:
        from google.genai.errors import APIError as GoogleAPIError

        error_types.append(GoogleAPIError)
    except ImportError:  # pragma: no cover - google extra is installed
        pass

    return tuple(error_types)


_MODEL_ERROR_TYPES: tuple[type[BaseException], ...] = _collect_model_error_types()


def _get_media_type(file_path: str) -> str:
    """Return the MIME type for a file path (e.g. 'application/pdf')."""
    media_type, _ = mimetypes.guess_type(file_path)
    return media_type or _DEFAULT_MEDIA_TYPE


def _get_media(file_path: str) -> tuple[bytes, str]:
    """Read media bytes from a local path or http(s) URL and return (bytes, media_type)."""
    if file_path.startswith(_URL_PREFIXES):
        response = httpx.get(file_path, follow_redirects=True, timeout=_URL_FETCH_TIMEOUT)
        response.raise_for_status()
        media_bytes = response.content
        media_type = _get_media_type(file_path)
        # If the URL extension didn't tell us anything, trust the server's Content-Type.
        if media_type == _DEFAULT_MEDIA_TYPE:
            header = response.headers.get("content-type", "").split(";", 1)[0].strip()
            if header:
                media_type = header
    else:
        media_bytes = Path(file_path).read_bytes()
        media_type = _get_media_type(file_path)

    return media_bytes, media_type


def extract(schema: type[T], model: str, input_file: str, instructions: str | None = None) -> T:
    """
    Extract structured data from a document, image, audio, or video file using an LLM.

    Args:
        schema: A Pydantic model class defining the expected output structure.
        model: The model identifier (e.g., 'openai:gpt-5').
        input_file: A local file path or an http(s) URL to extract from.
        instructions: Optional natural-language guidance for the LLM.

    Returns:
        An instance of the schema populated with extracted data.

    Raises:
        UrlFetchError: If the URL cannot be fetched or returns a non-2xx status.
        SchemaValidationError: If the model output doesn't match the schema.
        ModelError: If there's an error communicating with the model API.
        ExtractionError: For other extraction failures.
    """
    try:
        load_dotenv()
        file_bytes, file_type = _get_media(file_path=input_file)
        agent = Agent(
            model,
            instructions=instructions,
            output_type=NativeOutput(schema) if model.startswith("ollama") else schema,
        )
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
        if isinstance(e, _MODEL_ERROR_TYPES):
            raise ModelError(f"Model API error: {e}") from e
        raise ExtractionError(f"Extraction failed: {e}") from e
