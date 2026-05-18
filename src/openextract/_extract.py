"""Core extraction functionality."""

import asyncio
import ipaddress
import mimetypes
import os
import random
import socket
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TypeVar
from urllib.parse import urlparse

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
_MAX_REDIRECTS = 10
_ALLOW_PRIVATE_URLS_ENV = "OPENEXTRACT_ALLOW_PRIVATE_URLS"
_BYTES_MEDIA_TYPE_REQUIRED = (
    "media_type is required when input_file is bytes or a file-like object; "
    "pass it explicitly, e.g. extract(..., media_type='application/pdf')."
)


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
        # Also covers OpenRouter, which uses the openai SDK under the hood.
        from openai import APIError as OpenAIAPIError

        error_types.append(OpenAIAPIError)
    except ImportError:  # pragma: no cover - openai extra is installed
        pass

    try:
        from anthropic import APIError as AnthropicAPIError

        error_types.append(AnthropicAPIError)
    except ImportError:  # pragma: no cover - anthropic extra is installed
        pass

    try:
        from google.genai.errors import APIError as GoogleAPIError

        error_types.append(GoogleAPIError)
    except ImportError:  # pragma: no cover - google extra is installed
        pass

    try:
        from botocore.exceptions import ClientError as BedrockClientError

        error_types.append(BedrockClientError)
    except ImportError:  # pragma: no cover - bedrock extra is installed
        pass

    try:
        from cohere.core.api_error import ApiError as CohereApiError

        error_types.append(CohereApiError)
    except ImportError:  # pragma: no cover - cohere extra is installed
        pass

    try:
        from huggingface_hub.errors import HfHubHTTPError

        error_types.append(HfHubHTTPError)
    except ImportError:  # pragma: no cover - huggingface extra is installed
        pass

    try:
        from groq import APIError as GroqAPIError

        error_types.append(GroqAPIError)
    except ImportError:  # pragma: no cover - groq extra is installed
        pass

    try:
        from mistralai.client.errors.mistralerror import MistralError

        error_types.append(MistralError)
    except ImportError:  # pragma: no cover - mistral extra is installed
        pass

    return tuple(error_types)


_MODEL_ERROR_TYPES: tuple[type[BaseException], ...] = _collect_model_error_types()


@dataclass(frozen=True)
class Usage:
    """Token usage information for a single extraction call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


def _get_media_type(file_path: str) -> str:
    """Return the MIME type for a file path (e.g. 'application/pdf')."""
    media_type, _ = mimetypes.guess_type(file_path)
    return media_type or _DEFAULT_MEDIA_TYPE


def _allow_private_urls() -> bool:
    """Return True when SSRF host validation is disabled via env var."""
    return os.environ.get(_ALLOW_PRIVATE_URLS_ENV, "").lower() in ("1", "true", "yes")


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True only for globally routable public unicast addresses.

    Treats IPv4-mapped IPv6 (``::ffff:a.b.c.d``) as its underlying IPv4 so
    that e.g. ``::ffff:127.0.0.1`` is correctly classified as loopback.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global


def _is_safe_host(host: str | None) -> bool:
    """Return True when ``host`` resolves only to public IP addresses.

    Refuses missing/empty hosts, IP literals in private/loopback/link-local/
    multicast/reserved ranges (including AWS/GCP metadata at
    ``169.254.169.254``), hostnames that resolve to any non-public IP, and
    hostnames that fail to resolve. Bypassed by ``OPENEXTRACT_ALLOW_PRIVATE_URLS``.
    """
    if _allow_private_urls():
        return True
    if not host:
        return False
    bare = host.strip("[]")
    try:
        return _is_public_ip(ipaddress.ip_address(bare))
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(bare, None)
    except OSError:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            resolved = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if not _is_public_ip(resolved):
            return False
    return True


def _fetch_url(url: str) -> httpx.Response:
    """Fetch ``url`` with SSRF defenses; validate the host at every redirect hop."""
    current = url
    for _ in range(_MAX_REDIRECTS):
        host = urlparse(current).hostname
        if not _is_safe_host(host):
            raise UrlFetchError(f"Refusing to fetch URL with non-public host: {host!r}")
        response = httpx.get(current, follow_redirects=False, timeout=_URL_FETCH_TIMEOUT)
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise UrlFetchError(f"Redirect from {current!r} missing Location header")
            current = str(httpx.URL(current).join(location))
            continue
        response.raise_for_status()
        return response
    raise UrlFetchError(f"Too many redirects (>{_MAX_REDIRECTS})")


def _read_from_path(file_path: str) -> tuple[bytes, str]:
    """Read bytes from a local path or http(s) URL; return (bytes, media_type)."""
    if file_path.startswith(_URL_PREFIXES):
        response = _fetch_url(file_path)
        media_bytes = response.content
        media_type = _get_media_type(file_path)
        # If the URL extension didn't tell us anything, trust the server's Content-Type.
        if media_type == _DEFAULT_MEDIA_TYPE:
            header = response.headers.get("content-type", "").split(";", 1)[0].strip()
            if header:
                media_type = header
        return media_bytes, media_type

    return Path(file_path).read_bytes(), _get_media_type(file_path)


def _get_media(
    input_file: str | bytes | BinaryIO,
    media_type: str | None = None,
) -> tuple[bytes, str]:
    """Resolve ``input_file`` to ``(bytes, media_type)``.

    ``str`` is treated as a local path or http(s) URL. ``bytes`` and file-like
    objects (anything with a ``.read()`` method) are passed through. For the
    latter two, ``media_type`` is required.
    """
    if isinstance(input_file, str):
        file_bytes, resolved_type = _read_from_path(input_file)
        return file_bytes, media_type or resolved_type

    if isinstance(input_file, bytes):
        if media_type is None:
            raise TypeError(_BYTES_MEDIA_TYPE_REQUIRED)
        return input_file, media_type

    if hasattr(input_file, "read"):
        if media_type is None:
            raise TypeError(_BYTES_MEDIA_TYPE_REQUIRED)
        return input_file.read(), media_type

    raise TypeError(
        "input_file must be a str path/URL, bytes, or a file-like object with a .read() method."
    )


def _build_agent(schema: type[T], model: str, instructions: str | None) -> Agent:
    """Construct the pydantic_ai Agent, handling the ollama output-type quirk."""
    return Agent(
        model,
        instructions=instructions,
        output_type=NativeOutput(schema) if model.startswith("ollama") else schema,
    )


def _build_run_inputs(file_bytes: bytes, file_type: str) -> list:
    """Build the prompt inputs passed to the agent run."""
    return [
        "Extract the requested information from this document.",
        BinaryContent(data=file_bytes, media_type=file_type),
    ]


def _map_exception(exc: BaseException) -> ExtractionError:
    """Translate a low-level exception into the appropriate ExtractionError subclass."""
    if isinstance(exc, httpx.HTTPStatusError):
        return UrlFetchError(f"Failed to fetch URL: {exc.response.status_code}")
    if isinstance(exc, httpx.RequestError):
        return UrlFetchError(f"Failed to fetch URL: {exc}")
    if isinstance(exc, ValidationError):
        return SchemaValidationError(f"Model output did not match schema: {exc}")
    if isinstance(exc, _MODEL_ERROR_TYPES):
        return ModelError(f"Model API error: {exc}")
    return ExtractionError(f"Extraction failed: {exc}")


def _usage_from_result(result) -> Usage:
    """Build a ``Usage`` from a pydantic-ai run result."""
    raw = result.usage()
    return Usage(
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
    )


def _run_extraction(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
):
    """Run a single sync extraction and return the raw pydantic-ai result.

    Centralises agent build, exception mapping, and TypeError pass-through so it
    can be reused by ``extract`` (which discards usage) and
    ``extract_with_usage`` (which surfaces it).
    """
    try:
        load_dotenv()
        file_bytes, file_type = _get_media(input_file, media_type=media_type)
        agent = _build_agent(schema, model, instructions)
        return agent.run_sync(_build_run_inputs(file_bytes, file_type))
    except TypeError:
        raise
    except ExtractionError:
        raise
    except Exception as e:
        raise _map_exception(e) from e


def _extract_once(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
) -> T:
    """Perform a single sync extraction attempt; return the schema instance."""
    return _run_extraction(schema, model, input_file, instructions, media_type).output


def extract(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
) -> T:
    """
    Extract structured data from a document, image, audio, or video file using an LLM.

    Args:
        schema: A Pydantic model class defining the expected output structure.
        model: The model identifier (e.g., 'openai:gpt-5').
        input_file: A local file path, an ``http(s)://`` URL, raw ``bytes``, or
            a binary file-like object with a ``.read()`` method. For ``bytes``
            and file-like inputs, ``media_type`` must be provided.
        instructions: Optional natural-language guidance for the LLM.
        media_type: Optional MIME type. Required for ``bytes`` and file-like
            inputs; overrides the guess for ``str`` inputs when provided.
        max_retries: Number of additional attempts to make after a ``ModelError``.
            Defaults to 0 (no retries, single attempt). Only ``ModelError``
            triggers a retry; other exceptions propagate immediately.
        retry_backoff: Base backoff in seconds. Sleep between attempts is
            ``retry_backoff * (2 ** attempt) * (1 + random.uniform(0, 0.25))``,
            i.e. exponential backoff with up to 25% jitter.

    Returns:
        An instance of the schema populated with extracted data.

    Raises:
        TypeError: If ``input_file`` is bytes or file-like and ``media_type``
            is not provided.
        UrlFetchError: If the URL cannot be fetched or returns a non-2xx status.
        SchemaValidationError: If the model output doesn't match the schema.
        ModelError: If retries (if any) are exhausted.
        ExtractionError: For other extraction failures.
    """
    attempt = 0
    while True:
        try:
            return _extract_once(schema, model, input_file, instructions, media_type)
        except ModelError:
            if attempt >= max_retries:
                raise
            delay = retry_backoff * (2**attempt) * (1 + random.uniform(0, 0.25))
            time.sleep(delay)
            attempt += 1


def extract_with_usage(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
) -> tuple[T, Usage]:
    """Extract structured data and return ``(output, Usage)`` for token accounting.

    Behaves identically to :func:`extract` (without retry) but additionally
    returns a :class:`Usage` describing the tokens consumed by the model call.
    """
    result = _run_extraction(schema, model, input_file, instructions, media_type)
    return result.output, _usage_from_result(result)


async def extract_async(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
) -> T:
    """Async sibling of :func:`extract`; uses ``Agent.run`` instead of ``run_sync``."""
    try:
        load_dotenv()
        file_bytes, file_type = _get_media(input_file, media_type=media_type)
        agent = _build_agent(schema, model, instructions)
        result = await agent.run(_build_run_inputs(file_bytes, file_type))
        return result.output
    except TypeError:
        raise
    except ExtractionError:
        raise
    except Exception as e:
        raise _map_exception(e) from e


async def _gather_extractions(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None,
    max_concurrency: int,
    return_exceptions: bool,
) -> list:
    files = list(input_files)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(item):
        async with semaphore:
            return await extract_async(schema, model, item, instructions)

    tasks = [_bounded(item) for item in files]
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


def extract_many(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None = None,
    *,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
) -> list:
    """Run :func:`extract_async` over many inputs concurrently from sync code.

    Args:
        schema: A Pydantic model class defining the expected output structure.
        model: The model identifier.
        input_files: Iterable of paths, URLs, or already-resolved bytes.
        instructions: Optional natural-language guidance.
        max_concurrency: Maximum number of in-flight extractions.
        return_exceptions: If True, exceptions are returned in-place instead of raised
            (mirrors :func:`asyncio.gather`).

    Returns:
        A list of results (or exceptions, when ``return_exceptions=True``) in input order.
    """
    return asyncio.run(
        _gather_extractions(
            schema,
            model,
            input_files,
            instructions,
            max_concurrency,
            return_exceptions,
        )
    )


async def extract_many_async(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None = None,
    *,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
) -> list:
    """Async sibling of :func:`extract_many`."""
    return await _gather_extractions(
        schema,
        model,
        input_files,
        instructions,
        max_concurrency,
        return_exceptions,
    )
