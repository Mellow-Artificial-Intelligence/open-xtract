"""Core extraction functionality."""

import asyncio
import importlib
import ipaddress
import math
import mimetypes
import os
import random
import socket
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TypeVar, cast
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.output import NativeOutput

from .exceptions import (
    ExtractionError,
    InputTooLargeError,
    ModelError,
    ProviderNotInstalledError,
    SchemaValidationError,
    UrlFetchError,
)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MEDIA_TYPE = "application/octet-stream"
_URL_PREFIXES = ("http://", "https://")
_DEFAULT_URL_FETCH_TIMEOUT = 30.0
_DEFAULT_MAX_REDIRECTS = 10
_DEFAULT_MAX_INPUT_BYTES = 52_428_800  # 50 MiB
_URL_TIMEOUT_ENV = "OPENEXTRACT_URL_TIMEOUT"
_MAX_REDIRECTS_ENV = "OPENEXTRACT_MAX_REDIRECTS"
_ALLOW_PRIVATE_URLS_ENV = "OPENEXTRACT_ALLOW_PRIVATE_URLS"
_MAX_INPUT_BYTES_ENV = "OPENEXTRACT_MAX_INPUT_BYTES"
_READ_CHUNK_SIZE = 65_536
_BYTES_MEDIA_TYPE_REQUIRED = (
    "media_type is required when input_file is bytes or a file-like object; "
    "pass it explicitly, e.g. extract(..., media_type='application/pdf')."
)

# (module, attribute) pairs for provider/model error base classes we want to
# classify as ``ModelError``. ``openai.APIError`` also covers OpenRouter and
# Cerebras since both go through the openai SDK.
_PROVIDER_ERROR_PATHS: tuple[tuple[str, str], ...] = (
    ("pydantic_ai.exceptions", "ModelAPIError"),
    ("openai", "APIError"),
    ("anthropic", "APIError"),
    ("google.genai.errors", "APIError"),
    ("botocore.exceptions", "ClientError"),
    ("cohere.core.api_error", "ApiError"),
    ("huggingface_hub.errors", "HfHubHTTPError"),
    ("groq", "APIError"),
    ("mistralai.client.errors.mistralerror", "MistralError"),
    ("grpc", "RpcError"),  # xAI SDK uses gRPC; pydantic-ai may surface this directly
)

# pydantic-ai model-id prefix -> the openextract optional-dependency extra that
# installs that provider's SDK. OpenAI-compatible providers (Cerebras, Ollama)
# ship through the openai SDK, so they map to the ``openai`` extra. Prefixes not
# listed here fall back to suggesting ``openextract[all]``.
_PROVIDER_EXTRAS: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google-gla": "google",
    "google-vertex": "google",
    "bedrock": "bedrock",
    "cohere": "cohere",
    "groq": "groq",
    "huggingface": "huggingface",
    "mistral": "mistral",
    "openrouter": "openrouter",
    "xai": "xai",
    "cerebras": "openai",
    "ollama": "openai",
}


def _collect_model_error_types() -> tuple[type[BaseException], ...]:
    """Resolve ``_PROVIDER_ERROR_PATHS`` to importable exception classes.

    Each import is guarded so a missing optional provider does not break the
    package. The returned tuple is suitable for use with ``isinstance``.

    Cached after the first call. Deferred (rather than run at module import)
    because importing every provider SDK eagerly added ~2 s to cold start —
    ``google.genai`` alone is over a second — and is wasted work unless we
    actually need to classify an exception.
    """
    global _MODEL_ERROR_TYPES
    if _MODEL_ERROR_TYPES is not None:
        return _MODEL_ERROR_TYPES

    error_types: list[type[BaseException]] = []
    for module_name, attr in _PROVIDER_ERROR_PATHS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover - all provider extras are installed
            continue
        error_types.append(getattr(module, attr))
    _MODEL_ERROR_TYPES = tuple(error_types)
    return _MODEL_ERROR_TYPES


_MODEL_ERROR_TYPES: tuple[type[BaseException], ...] | None = None


_DOTENV_LOADED = False


def _ensure_dotenv_loaded() -> None:
    """Run ``load_dotenv()`` at most once per process.

    Re-scanning the filesystem on every extract added ~50 µs/call and changed
    nothing — the loaded env vars persist after the first call.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    load_dotenv()
    _DOTENV_LOADED = True


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


def _env_positive_float(name: str, default: float) -> float:
    """Parse a positive float from ``name``; return ``default`` when unset or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_positive_int(name: str, default: int) -> int:
    """Parse a positive int from ``name``; return ``default`` when unset or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _url_fetch_timeout() -> float:
    """HTTP timeout in seconds for URL fetches (``OPENEXTRACT_URL_TIMEOUT``)."""
    return _env_positive_float(_URL_TIMEOUT_ENV, _DEFAULT_URL_FETCH_TIMEOUT)


def _max_redirects() -> int:
    """Maximum redirect hops when fetching URLs (``OPENEXTRACT_MAX_REDIRECTS``)."""
    return _env_positive_int(_MAX_REDIRECTS_ENV, _DEFAULT_MAX_REDIRECTS)


def _too_large_message(limit: int, got: int) -> str:
    return (
        f"Input exceeds the configured size limit ({limit} bytes); "
        f"got at least {got} bytes. "
        f"Set {_MAX_INPUT_BYTES_ENV} or pass max_input_bytes=... if this is intentional."
    )


def _resolve_max_input_bytes(max_input_bytes: int | None) -> int:
    """Resolve the input size limit from kwarg, env, or the 50 MiB default."""
    if max_input_bytes is not None:
        if (
            isinstance(max_input_bytes, bool)
            or not isinstance(max_input_bytes, int)
            or max_input_bytes <= 0
        ):
            raise ValueError("max_input_bytes must be a positive integer.")
        return max_input_bytes

    raw = os.environ.get(_MAX_INPUT_BYTES_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_INPUT_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_MAX_INPUT_BYTES_ENV} must be a positive integer, got {raw!r}."
        ) from exc
    if value <= 0:
        raise ValueError(f"{_MAX_INPUT_BYTES_ENV} must be a positive integer, got {value}.")
    return value


def _enforce_max_input_bytes(data: bytes, *, limit: int) -> bytes:
    """Return ``data`` or raise :class:`InputTooLargeError` when over ``limit``."""
    if len(data) > limit:
        raise InputTooLargeError(_too_large_message(limit, len(data)))
    return data


def _check_content_length(headers: httpx.Headers | dict[str, str], limit: int) -> None:
    """Fail fast when a trustworthy ``Content-Length`` exceeds ``limit``."""
    raw = headers.get("content-length")
    if raw is None:
        return
    try:
        length = int(str(raw).strip())
    except (TypeError, ValueError):
        return
    if length > limit:
        raise InputTooLargeError(_too_large_message(limit, length))


def _read_file_like_capped(file_obj: BinaryIO, *, limit: int) -> bytes:
    """Read a file-like object in chunks; stop when size exceeds ``limit``."""
    chunks: list[bytes] = []
    total = 0
    while True:
        # Read one byte past the limit so we can detect oversize without seeking.
        chunk = file_obj.read(min(_READ_CHUNK_SIZE, limit - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise InputTooLargeError(_too_large_message(limit, total))
        chunks.append(chunk)
    return b"".join(chunks)


def _accumulate_capped(chunks: Iterable[bytes], *, limit: int) -> bytes:
    """Join ``chunks`` until empty, raising if the total exceeds ``limit``."""
    parts: list[bytes] = []
    total = 0
    for chunk in chunks:
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise InputTooLargeError(_too_large_message(limit, total))
        parts.append(chunk)
    return b"".join(parts)


def _read_response_body_capped(response: httpx.Response, *, limit: int) -> bytes:
    """Read an HTTP response body with a hard size cap.

    Real ``httpx.Response`` objects are consumed via ``iter_bytes`` so a streamed
    response can fail before buffering the full payload. Mocked responses fall
    back to ``response.content``.
    """
    _check_content_length(response.headers, limit)
    if isinstance(response, httpx.Response):
        return _accumulate_capped(response.iter_bytes(), limit=limit)
    return _enforce_max_input_bytes(response.content, limit=limit)


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
    limit = _max_redirects()
    timeout = _url_fetch_timeout()
    for _ in range(limit):
        host = urlparse(current).hostname
        if not _is_safe_host(host):
            raise UrlFetchError(f"Refusing to fetch URL with non-public host: {host!r}")
        response = httpx.get(current, follow_redirects=False, timeout=timeout)
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise UrlFetchError(f"Redirect from {current!r} missing Location header")
            current = str(httpx.URL(current).join(location))
            continue
        response.raise_for_status()
        return response
    raise UrlFetchError(f"Too many redirects (>{limit})")


def _fetch_url_bytes(url: str, *, max_bytes: int) -> tuple[bytes, httpx.Headers]:
    """Fetch ``url`` and return ``(body, headers)`` capped to ``max_bytes``."""
    response = _fetch_url(url)
    return _read_response_body_capped(response, limit=max_bytes), response.headers


def _read_from_path(file_path: str, *, max_bytes: int) -> tuple[bytes, str]:
    """Read bytes from a local path or http(s) URL; return (bytes, media_type)."""
    if file_path.startswith(_URL_PREFIXES):
        media_bytes, headers = _fetch_url_bytes(file_path, max_bytes=max_bytes)
        media_type = _get_media_type(file_path)
        # If the URL extension didn't tell us anything, trust the server's Content-Type.
        if media_type == _DEFAULT_MEDIA_TYPE:
            header = headers.get("content-type", "").split(";", 1)[0].strip()
            if header:
                media_type = header
        return media_bytes, media_type

    path = Path(file_path)
    size = path.stat().st_size
    if size > max_bytes:
        raise InputTooLargeError(_too_large_message(max_bytes, size))
    return _enforce_max_input_bytes(path.read_bytes(), limit=max_bytes), _get_media_type(file_path)


def _get_media(
    input_file: str | bytes | BinaryIO,
    media_type: str | None = None,
    *,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    """Resolve ``input_file`` to ``(bytes, media_type)``.

    ``str`` is treated as a local path or http(s) URL. ``bytes`` and file-like
    objects (anything with a ``.read()`` method) are passed through. For the
    latter two, ``media_type`` is required.
    """
    limit = _resolve_max_input_bytes(max_bytes)

    if isinstance(input_file, str):
        file_bytes, resolved_type = _read_from_path(input_file, max_bytes=limit)
        return file_bytes, media_type or resolved_type

    if isinstance(input_file, bytes):
        if media_type is None:
            raise TypeError(_BYTES_MEDIA_TYPE_REQUIRED)
        return _enforce_max_input_bytes(input_file, limit=limit), media_type

    if hasattr(input_file, "read"):
        if media_type is None:
            raise TypeError(_BYTES_MEDIA_TYPE_REQUIRED)
        return _read_file_like_capped(input_file, limit=limit), media_type

    raise TypeError(
        "input_file must be a str path/URL, bytes, or a file-like object with a .read() method."
    )


def _install_hint(model: str) -> str:
    """Return the ``pip install`` command that provides ``model``'s provider."""
    prefix = model.split(":", 1)[0]
    extra = _PROVIDER_EXTRAS.get(prefix)
    target = f"openextract[{extra}]" if extra else "openextract[all]"
    return f"pip install '{target}'"


def _build_agent(schema: type[T], model: str, instructions: str | None) -> Agent:
    """Construct the pydantic_ai Agent, handling the ollama output-type quirk.

    A missing provider SDK surfaces here as ``ImportError`` (pydantic-ai infers
    the model eagerly); translate it into an actionable
    :class:`ProviderNotInstalledError`.
    """
    try:
        return Agent(
            model,
            instructions=instructions,
            output_type=NativeOutput(schema) if model.startswith("ollama") else schema,
        )
    except ImportError as exc:
        raise ProviderNotInstalledError(
            f"Model {model!r} needs a provider SDK that is not installed. "
            f"Install it with: {_install_hint(model)} "
            f"(or 'pip install openextract[all]'). Original error: {exc}"
        ) from exc


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
    if isinstance(exc, _collect_model_error_types()):
        return ModelError(f"Model API error: {exc}")
    return ExtractionError(f"Extraction failed: {exc}")


@contextmanager
def _extraction_errors() -> Iterator[None]:
    """Map low-level extraction failures to the ``ExtractionError`` hierarchy.

    ``TypeError`` (bad call) and already-mapped ``ExtractionError`` subclasses
    pass through unchanged; everything else is routed through
    :func:`_map_exception`.
    """
    try:
        yield
    except (TypeError, ExtractionError):
        raise
    except Exception as e:
        raise _map_exception(e) from e


def _usage_from_result(result) -> Usage:
    """Build a ``Usage`` from a pydantic-ai run result."""
    raw = result.usage()
    return Usage(
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
    )


def _prepare_run(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int | None = None,
) -> tuple[Agent, list]:
    """Load env, resolve the media payload, and build the agent + run inputs."""
    _ensure_dotenv_loaded()
    file_bytes, file_type = _get_media(input_file, media_type=media_type, max_bytes=max_input_bytes)
    agent = _build_agent(schema, model, instructions)
    return agent, _build_run_inputs(file_bytes, file_type)


def _run_extraction(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int | None = None,
):
    """Run a single sync extraction and return the raw pydantic-ai result.

    Centralises agent build, exception mapping, and TypeError pass-through so it
    can be reused by ``extract`` (which discards usage) and
    ``extract_with_usage`` (which surfaces it).
    """
    with _extraction_errors():
        agent, inputs = _prepare_run(
            schema, model, input_file, instructions, media_type, max_input_bytes
        )
        return agent.run_sync(inputs)


def _extract_once(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int | None = None,
) -> T:
    """Perform a single sync extraction attempt; return the schema instance."""
    result = _run_extraction(schema, model, input_file, instructions, media_type, max_input_bytes)
    return cast(T, result.output)


def _retry_delay(retry_backoff: float, attempt: int) -> float:
    return retry_backoff * (2**attempt) * (1 + random.uniform(0, 0.25))


def _validate_retry_options(max_retries: object, retry_backoff: object) -> None:
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")
    if (
        isinstance(retry_backoff, bool)
        or not isinstance(retry_backoff, int | float)
        or not math.isfinite(retry_backoff)
        or retry_backoff <= 0
    ):
        raise ValueError("retry_backoff must be a finite positive number of seconds.")


def _validate_max_concurrency(max_concurrency: object) -> None:
    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency < 1
    ):
        raise ValueError("max_concurrency must be a positive integer.")


def _run_with_retries_sync[R](
    fn: Callable[[], R],
    *,
    max_retries: int,
    retry_backoff: float,
) -> R:
    """Run ``fn`` until it succeeds or ``ModelError`` retries are exhausted."""
    _validate_retry_options(max_retries, retry_backoff)
    attempt = 0
    while True:
        try:
            return fn()
        except ModelError:
            if attempt >= max_retries:
                raise
            time.sleep(_retry_delay(retry_backoff, attempt))
            attempt += 1


async def _run_with_retries_async[R](
    fn: Callable[[], Awaitable[R]],
    *,
    max_retries: int,
    retry_backoff: float,
) -> R:
    """Async counterpart to :func:`_run_with_retries_sync`."""
    _validate_retry_options(max_retries, retry_backoff)
    attempt = 0
    while True:
        try:
            return await fn()
        except ModelError:
            if attempt >= max_retries:
                raise
            await asyncio.sleep(_retry_delay(retry_backoff, attempt))
            attempt += 1


def extract(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    max_input_bytes: int | None = None,
) -> T:
    """
    Extract structured data from a document, image, audio, or video file using an LLM.

    Args:
        schema: A Pydantic model class defining the expected output structure.
        model: The model identifier (e.g., 'xai:grok-4.3').
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
        max_input_bytes: Optional hard cap on resolved input size. ``None`` uses
            ``OPENEXTRACT_MAX_INPUT_BYTES`` or the default 50 MiB. Must be a
            positive integer when set.

    Returns:
        An instance of the schema populated with extracted data.

    Raises:
        TypeError: If ``input_file`` is bytes or file-like and ``media_type``
            is not provided.
        ValueError: If ``max_input_bytes`` / env override is not a positive int.
        InputTooLargeError: If the resolved input exceeds the configured size limit.
        UrlFetchError: If the URL cannot be fetched or returns a non-2xx status.
        SchemaValidationError: If the model output doesn't match the schema.
        ModelError: If retries (if any) are exhausted.
        ExtractionError: For other extraction failures.
    """
    limit = _resolve_max_input_bytes(max_input_bytes)
    return _run_with_retries_sync(
        lambda: _extract_once(schema, model, input_file, instructions, media_type, limit),
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )


def extract_with_usage(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    max_input_bytes: int | None = None,
) -> tuple[T, Usage]:
    """Extract structured data and return ``(output, Usage)`` for token accounting.

    Same retry semantics as :func:`extract`. Returns a :class:`Usage` describing
    the tokens consumed by the successful model call.
    """
    limit = _resolve_max_input_bytes(max_input_bytes)

    def _once() -> tuple[T, Usage]:
        result = _run_extraction(schema, model, input_file, instructions, media_type, limit)
        return cast(T, result.output), _usage_from_result(result)

    return _run_with_retries_sync(_once, max_retries=max_retries, retry_backoff=retry_backoff)


async def extract_with_usage_async(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    max_input_bytes: int | None = None,
) -> tuple[T, Usage]:
    """Async sibling of :func:`extract_with_usage`; returns ``(output, Usage)``."""
    limit = _resolve_max_input_bytes(max_input_bytes)

    async def _once() -> tuple[T, Usage]:
        with _extraction_errors():
            agent, inputs = _prepare_run(schema, model, input_file, instructions, media_type, limit)
            result = await agent.run(inputs)
            return cast(T, result.output), _usage_from_result(result)

    return await _run_with_retries_async(
        _once,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )


async def extract_async(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    max_input_bytes: int | None = None,
) -> T:
    """Async sibling of :func:`extract`; uses ``Agent.run`` instead of ``run_sync``."""
    limit = _resolve_max_input_bytes(max_input_bytes)

    async def _once() -> T:
        with _extraction_errors():
            agent, inputs = _prepare_run(schema, model, input_file, instructions, media_type, limit)
            result = await agent.run(inputs)
            return cast(T, result.output)

    return await _run_with_retries_async(
        _once,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )


async def _run_with_shared_agent(
    agent: Agent,
    input_file: str | bytes | BinaryIO,
    media_type: str | None,
    max_input_bytes: int | None = None,
) -> object:
    """Run a single extraction reusing a pre-built ``Agent``.

    Mirrors ``extract_async``'s error mapping so callers get the same
    ``ExtractionError`` subclasses as the per-item path.
    """
    limit = _resolve_max_input_bytes(max_input_bytes)
    with _extraction_errors():
        file_bytes, file_type = _get_media(input_file, media_type=media_type, max_bytes=limit)
        result = await agent.run(_build_run_inputs(file_bytes, file_type))
        return result.output


async def _gather_extractions(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None,
    max_concurrency: int,
    return_exceptions: bool,
    media_type: str | None,
    max_retries: int,
    retry_backoff: float,
    max_input_bytes: int | None = None,
) -> list:
    _validate_retry_options(max_retries, retry_backoff)
    _validate_max_concurrency(max_concurrency)
    limit = _resolve_max_input_bytes(max_input_bytes)
    files = list(input_files)
    if not files:
        return []
    _ensure_dotenv_loaded()
    # Building the Agent (and its provider HTTP client) is ~32 ms; sharing one
    # across the batch saves ~32 ms × (N-1) per call. The Agent is stateless
    # between runs and stays inside this event loop, so this is safe.
    agent = _build_agent(schema, model, instructions)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(item):
        async def _once():
            return await _run_with_shared_agent(agent, item, media_type, limit)

        async with semaphore:
            return await _run_with_retries_async(
                _once,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
            )

    tasks = [_bounded(item) for item in files]
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


def extract_many(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    max_input_bytes: int | None = None,
) -> list:
    """Run :func:`extract_async` over many inputs concurrently from sync code.

    Args:
        schema: A Pydantic model class defining the expected output structure.
        model: The model identifier.
        input_files: Iterable of paths, URLs, or already-resolved bytes.
        instructions: Optional natural-language guidance.
        media_type: Optional MIME type applied uniformly to every item.  Required
            when ``input_files`` contains ``bytes`` or file-like objects; optional
            override for path/URL items.
        max_concurrency: Maximum number of in-flight extractions.
        return_exceptions: If True, exceptions are returned in-place instead of raised
            (mirrors :func:`asyncio.gather`).
        max_retries: Per-item retries after ``ModelError`` (same semantics as
            :func:`extract`).
        retry_backoff: Base backoff seconds between per-item retries.
        max_input_bytes: Optional hard cap on each item's resolved input size.
            ``None`` uses ``OPENEXTRACT_MAX_INPUT_BYTES`` or the default 50 MiB.

    Returns:
        A list of results (or exceptions, when ``return_exceptions=True``) in input order.

    Raises:
        ValueError: If ``max_concurrency`` is less than 1, ``max_retries`` is
            negative, ``retry_backoff`` is not positive and finite, or
            ``max_input_bytes`` is not a positive integer when set.
        RuntimeError: If called from a running event loop. Use
            :func:`extract_many_async` in async code instead.
    """
    _validate_retry_options(max_retries, retry_backoff)
    _validate_max_concurrency(max_concurrency)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "extract_many() cannot be called from a running event loop; "
            "use await extract_many_async(...) instead."
        )
    return asyncio.run(
        _gather_extractions(
            schema,
            model,
            input_files,
            instructions,
            max_concurrency,
            return_exceptions,
            media_type,
            max_retries,
            retry_backoff,
            max_input_bytes,
        )
    )


async def extract_many_async(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    max_input_bytes: int | None = None,
) -> list:
    """Async sibling of :func:`extract_many`."""
    return await _gather_extractions(
        schema,
        model,
        input_files,
        instructions,
        max_concurrency,
        return_exceptions,
        media_type,
        max_retries,
        retry_backoff,
        max_input_bytes,
    )
