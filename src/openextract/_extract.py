"""Core extraction functionality."""

import asyncio
import ipaddress
import math
import mimetypes
import os
import random
import socket
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, TypeVar, cast
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from pydantic_ai import Agent
else:

    def Agent(*args, **kwargs):
        """Construct a Pydantic AI agent without loading it at package import time."""
        from pydantic_ai import Agent as PydanticAgent

        return PydanticAgent(*args, **kwargs)


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
_DEFAULT_RETRY_MAX_BACKOFF = 60.0
_DEFAULT_MAX_INPUT_BYTES = 50 * 1024 * 1024
_INPUT_READ_CHUNK_SIZE = 64 * 1024
_OPENAI_PREFIX = "openai:"
_OPENAI_RESPONSES_PREFIX = "openai-responses:"
_URL_TIMEOUT_ENV = "OPENEXTRACT_URL_TIMEOUT"
_MAX_REDIRECTS_ENV = "OPENEXTRACT_MAX_REDIRECTS"
_ALLOW_PRIVATE_URLS_ENV = "OPENEXTRACT_ALLOW_PRIVATE_URLS"
_MAX_INPUT_BYTES_ENV = "OPENEXTRACT_MAX_INPUT_BYTES"
_BYTES_MEDIA_TYPE_REQUIRED = (
    "media_type is required when input_file is bytes or a file-like object; "
    "pass it explicitly, e.g. extract(..., media_type='application/pdf')."
)
_TRANSIENT_HTTP_STATUSES = frozenset((408, 409, 425, 429))
_PROVIDER_MODULE_NAMES: tuple[tuple[str, str], ...] = (
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("google", "google"),
    ("botocore", "bedrock"),
    ("cohere", "cohere"),
    ("huggingface_hub", "huggingface"),
    ("groq", "groq"),
    ("mistralai", "mistral"),
    ("grpc", "xai"),
)
_TRANSIENT_ERROR_NAMES = (
    "timeout",
    "connection",
    "ratelimit",
    "throttl",
    "resourceexhausted",
    "servererror",
    "internalserver",
    "serviceunavailable",
)
_PERMANENT_ERROR_NAMES = (
    "authentication",
    "permission",
    "forbidden",
    "badrequest",
    "invalidrequest",
    "validation",
    "unprocessable",
    "notfound",
    "accessdenied",
)

# Exact (module, class) signatures for provider/model error bases we classify
# as ``ModelError``. The classifier checks the exception's existing MRO instead
# of importing provider SDKs. ``openai.APIError`` also covers OpenRouter and
# Cerebras since both go through the openai SDK.
_MODEL_ERROR_SIGNATURES = frozenset(
    {
        ("pydantic_ai.exceptions", "ModelAPIError"),
        ("openai", "APIError"),
        ("anthropic", "APIError"),
        ("google.genai.errors", "APIError"),
        ("botocore.exceptions", "ClientError"),
        ("botocore.exceptions", "BotoCoreError"),
        ("cohere.core.api_error", "ApiError"),
        ("huggingface_hub.errors", "HfHubHTTPError"),
        ("groq", "APIError"),
        ("mistralai.client.errors.mistralerror", "MistralError"),
        ("grpc", "RpcError"),  # xAI SDK uses gRPC; pydantic-ai may surface this directly
    }
)

# pydantic-ai model-id prefix -> the openextract optional-dependency extra that
# installs that provider's SDK. OpenAI-compatible providers (Cerebras, Ollama)
# ship through the openai SDK, so they map to the ``openai`` extra. Prefixes not
# listed here fall back to suggesting ``openextract[all]``.
_PROVIDER_EXTRAS: dict[str, str] = {
    "openai": "openai",
    "openai-chat": "openai",
    "openai-responses": "openai",
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


def _is_model_exception(exc: BaseException) -> bool:
    """Return whether an exception inherits from a supported model error base.

    Provider SDKs have already created the exception by the time this runs, so
    their classes are present in its MRO. Comparing exact module/class
    signatures preserves subclass handling without importing unrelated SDKs.
    """
    return any(
        (error_type.__module__, error_type.__name__) in _MODEL_ERROR_SIGNATURES
        for error_type in type(exc).__mro__
    )


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


def _resolve_max_input_bytes(max_input_bytes: object) -> int:
    """Resolve and validate the per-input byte limit.

    An explicit value wins over ``OPENEXTRACT_MAX_INPUT_BYTES``. Invalid
    configured values fail closed instead of silently disabling the limit.
    """
    value = max_input_bytes
    from_environment = False
    if value is None:
        raw = os.environ.get(_MAX_INPUT_BYTES_ENV, "").strip()
        if not raw:
            return _DEFAULT_MAX_INPUT_BYTES
        from_environment = True
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{_MAX_INPUT_BYTES_ENV} must be a positive integer.") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        if from_environment:
            raise ValueError(f"{_MAX_INPUT_BYTES_ENV} must be a positive integer.")
        raise ValueError("max_input_bytes must be a positive integer.")
    return value


def _safe_source_context(source: str) -> str:
    """Return source context without URL credentials, query strings, or fragments."""
    if source.startswith(_URL_PREFIXES):
        parsed = urlparse(source)
        host = parsed.hostname or "unknown-host"
        try:
            parsed_port = parsed.port
        except ValueError:
            parsed_port = None
        port = f":{parsed_port}" if parsed_port is not None else ""
        return f"URL {parsed.scheme}://{host}{port}{parsed.path or '/'}"
    return f"path {Path(source).name!r}"


def _input_too_large(*, limit: int, observed: int, source: str) -> InputTooLargeError:
    return InputTooLargeError(
        f"{source} exceeds the configured size limit ({limit} bytes); "
        f"got at least {observed} bytes. Set {_MAX_INPUT_BYTES_ENV} or pass "
        "max_input_bytes=... if this is intentional."
    )


def _enforce_max_input_bytes(data: bytes, *, limit: int, source: str) -> bytes:
    if len(data) > limit:
        raise _input_too_large(limit=limit, observed=len(data), source=source)
    return data


def _read_file_like_limited(stream: BinaryIO, *, limit: int, source: str) -> bytes:
    """Read a binary stream in bounded chunks, including non-seekable streams."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(_INPUT_READ_CHUNK_SIZE, limit - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise _input_too_large(limit=limit, observed=total, source=source)
        chunks.append(chunk)
    return b"".join(chunks)


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
    with httpx.Client(follow_redirects=False, timeout=_url_fetch_timeout()) as client:
        return _fetch_url_with_client(url, client)


def _fetch_url_with_client(url: str, client: httpx.Client) -> httpx.Response:
    """Fetch ``url`` through ``client`` while validating every redirect target."""
    current = url
    limit = _max_redirects()
    for _ in range(limit):
        host = urlparse(current).hostname
        if not _is_safe_host(host):
            raise UrlFetchError(f"Refusing to fetch URL with non-public host: {host!r}")
        response = client.get(current)
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise UrlFetchError(f"Redirect from {current!r} missing Location header")
            current = str(httpx.URL(current).join(location))
            continue
        response.raise_for_status()
        return response
    raise UrlFetchError(f"Too many redirects (>{limit})")


async def _fetch_url_async(url: str, client: httpx.AsyncClient) -> httpx.Response:
    """Async URL fetch with the same redirect-by-redirect SSRF validation."""
    current = url
    limit = _max_redirects()
    for _ in range(limit):
        host = urlparse(current).hostname
        if not await asyncio.to_thread(_is_safe_host, host):
            raise UrlFetchError(f"Refusing to fetch URL with non-public host: {host!r}")
        response = await client.get(current)
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise UrlFetchError(f"Redirect from {current!r} missing Location header")
            current = str(httpx.URL(current).join(location))
            continue
        response.raise_for_status()
        return response
    raise UrlFetchError(f"Too many redirects (>{limit})")


def _response_content_length(response: httpx.Response) -> int | None:
    raw = response.headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _read_response_limited(
    response: httpx.Response,
    *,
    limit: int,
    source: str,
) -> bytes:
    content_length = _response_content_length(response)
    if content_length is not None and content_length > limit:
        raise _input_too_large(limit=limit, observed=content_length, source=source)

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes(chunk_size=_INPUT_READ_CHUNK_SIZE):
        total += len(chunk)
        if total > limit:
            raise _input_too_large(limit=limit, observed=total, source=source)
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_response_limited_async(
    response: httpx.Response,
    *,
    limit: int,
    source: str,
) -> bytes:
    content_length = _response_content_length(response)
    if content_length is not None and content_length > limit:
        raise _input_too_large(limit=limit, observed=content_length, source=source)

    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes(chunk_size=_INPUT_READ_CHUNK_SIZE):
        total += len(chunk)
        if total > limit:
            raise _input_too_large(limit=limit, observed=total, source=source)
        chunks.append(chunk)
    return b"".join(chunks)


def _read_url_with_client(
    url: str,
    client: httpx.Client,
    *,
    limit: int,
) -> tuple[bytes, Mapping[str, str]]:
    """Fetch a URL and stream its final response through the byte cap."""
    current = url
    redirect_limit = _max_redirects()
    for _ in range(redirect_limit):
        host = urlparse(current).hostname
        if not _is_safe_host(host):
            raise UrlFetchError(f"Refusing to fetch URL with non-public host: {host!r}")
        response = client.send(client.build_request("GET", current), stream=True)
        try:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise UrlFetchError(f"Redirect from {current!r} missing Location header")
                current = str(httpx.URL(current).join(location))
                continue
            response.raise_for_status()
            content = _read_response_limited(
                response,
                limit=limit,
                source=_safe_source_context(current),
            )
            return content, dict(response.headers)
        finally:
            response.close()
    raise UrlFetchError(f"Too many redirects (>{redirect_limit})")


async def _read_url_with_client_async(
    url: str,
    client: httpx.AsyncClient,
    *,
    limit: int,
) -> tuple[bytes, Mapping[str, str]]:
    """Async counterpart to :func:`_read_url_with_client`."""
    current = url
    redirect_limit = _max_redirects()
    for _ in range(redirect_limit):
        host = urlparse(current).hostname
        if not await asyncio.to_thread(_is_safe_host, host):
            raise UrlFetchError(f"Refusing to fetch URL with non-public host: {host!r}")
        response = await client.send(client.build_request("GET", current), stream=True)
        try:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise UrlFetchError(f"Redirect from {current!r} missing Location header")
                current = str(httpx.URL(current).join(location))
                continue
            response.raise_for_status()
            content = await _read_response_limited_async(
                response,
                limit=limit,
                source=_safe_source_context(current),
            )
            return content, dict(response.headers)
        finally:
            await response.aclose()
    raise UrlFetchError(f"Too many redirects (>{redirect_limit})")


def _read_url(url: str, *, limit: int) -> tuple[bytes, Mapping[str, str]]:
    with httpx.Client(follow_redirects=False, timeout=_url_fetch_timeout()) as client:
        return _read_url_with_client(url, client, limit=limit)


def _media_from_content(
    file_path: str,
    content: bytes,
    headers: Mapping[str, str],
) -> tuple[bytes, str]:
    media_type = _get_media_type(file_path)
    if media_type == _DEFAULT_MEDIA_TYPE:
        header = headers.get("content-type", "").split(";", 1)[0].strip()
        if header:
            media_type = header
    return content, media_type


def _read_from_path(file_path: str, *, max_input_bytes: int) -> tuple[bytes, str]:
    """Read bytes from a local path or http(s) URL; return (bytes, media_type)."""
    if file_path.startswith(_URL_PREFIXES):
        content, headers = _read_url(file_path, limit=max_input_bytes)
        return _media_from_content(file_path, content, headers)

    path = Path(file_path)
    source = _safe_source_context(file_path)
    size = path.stat().st_size
    if size > max_input_bytes:
        raise _input_too_large(limit=max_input_bytes, observed=size, source=source)
    with path.open("rb") as stream:
        content = _read_file_like_limited(stream, limit=max_input_bytes, source=source)
    return content, _get_media_type(file_path)


async def _read_from_path_async(
    file_path: str,
    client: httpx.AsyncClient | None,
    *,
    max_input_bytes: int,
) -> tuple[bytes, str]:
    """Async counterpart to :func:`_read_from_path`."""
    if file_path.startswith(_URL_PREFIXES):
        if client is None:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=_url_fetch_timeout(),
            ) as owned_client:
                content, headers = await _read_url_with_client_async(
                    file_path,
                    owned_client,
                    limit=max_input_bytes,
                )
        else:
            content, headers = await _read_url_with_client_async(
                file_path,
                client,
                limit=max_input_bytes,
            )
        return _media_from_content(file_path, content, headers)

    return await asyncio.to_thread(
        _read_from_path,
        file_path,
        max_input_bytes=max_input_bytes,
    )


def _get_media(
    input_file: str | bytes | BinaryIO,
    media_type: str | None = None,
    *,
    max_input_bytes: int | None = None,
) -> tuple[bytes, str]:
    """Resolve ``input_file`` to ``(bytes, media_type)``.

    ``str`` is treated as a local path or http(s) URL. ``bytes`` and file-like
    objects (anything with a ``.read()`` method) are passed through. For the
    latter two, ``media_type`` is required.
    """
    limit = _resolve_max_input_bytes(max_input_bytes)
    if isinstance(input_file, str):
        file_bytes, resolved_type = _read_from_path(input_file, max_input_bytes=limit)
        return file_bytes, media_type or resolved_type

    if isinstance(input_file, bytes):
        if media_type is None:
            raise TypeError(_BYTES_MEDIA_TYPE_REQUIRED)
        return (
            _enforce_max_input_bytes(input_file, limit=limit, source="bytes input"),
            media_type,
        )

    if hasattr(input_file, "read"):
        if media_type is None:
            raise TypeError(_BYTES_MEDIA_TYPE_REQUIRED)
        return (
            _read_file_like_limited(input_file, limit=limit, source="file-like input"),
            media_type,
        )

    raise TypeError(
        "input_file must be a str path/URL, bytes, or a file-like object with a .read() method."
    )


async def _get_media_async(
    input_file: str | bytes | BinaryIO,
    client: httpx.AsyncClient | None = None,
    media_type: str | None = None,
    *,
    max_input_bytes: int | None = None,
) -> tuple[bytes, str]:
    """Resolve media without blocking the event loop on disk, DNS, or stream I/O."""
    limit = _resolve_max_input_bytes(max_input_bytes)
    if isinstance(input_file, str):
        file_bytes, resolved_type = await _read_from_path_async(
            input_file,
            client,
            max_input_bytes=limit,
        )
        return file_bytes, media_type or resolved_type

    if isinstance(input_file, bytes):
        return _get_media(input_file, media_type=media_type, max_input_bytes=limit)

    if hasattr(input_file, "read"):
        return await asyncio.to_thread(
            _get_media,
            input_file,
            media_type,
            max_input_bytes=limit,
        )

    return _get_media(input_file, media_type=media_type, max_input_bytes=limit)


def _install_hint(model: str) -> str:
    """Return the ``pip install`` command that provides ``model``'s provider."""
    prefix = model.split(":", 1)[0]
    extra = _PROVIDER_EXTRAS.get(prefix)
    target = f"openextract[{extra}]" if extra else "openextract[all]"
    return f"pip install '{target}'"


def _route_model(model: str) -> str:
    """Route shorthand OpenAI identifiers through the Responses API."""
    if model.startswith(_OPENAI_PREFIX):
        return f"{_OPENAI_RESPONSES_PREFIX}{model.removeprefix(_OPENAI_PREFIX)}"
    return model


def _build_agent(schema: type[T], model: str, instructions: str | None) -> Agent:
    """Construct the pydantic_ai Agent, handling the ollama output-type quirk.

    A missing provider SDK surfaces here as ``ImportError`` (pydantic-ai infers
    the model eagerly); translate it into an actionable
    :class:`ProviderNotInstalledError`.
    """
    try:
        output_type = schema
        if model.startswith("ollama"):
            from pydantic_ai.output import NativeOutput

            output_type = NativeOutput(schema)
        return Agent(
            _route_model(model),
            instructions=instructions,
            output_type=output_type,
        )
    except ImportError as exc:
        raise ProviderNotInstalledError(
            f"Model {model!r} needs a provider SDK that is not installed. "
            f"Install it with: {_install_hint(model)} "
            f"(or 'pip install openextract[all]'). Original error: {exc}"
        ) from exc


def _build_run_inputs(file_bytes: bytes, file_type: str) -> list:
    """Build the prompt inputs passed to the agent run."""
    from pydantic_ai import BinaryContent

    return [
        "Extract the requested information from this document.",
        BinaryContent(data=file_bytes, media_type=file_type),
    ]


def _model_provider(exc: BaseException) -> str | None:
    """Return a stable provider identifier when the exception exposes one."""
    model_name = getattr(exc, "model_name", None)
    if isinstance(model_name, str) and ":" in model_name:
        return model_name.split(":", 1)[0]

    for error_type in type(exc).__mro__:
        module = error_type.__module__
        for prefix, provider in _PROVIDER_MODULE_NAMES:
            if module == prefix or module.startswith(f"{prefix}."):
                return provider
    return None


def _model_status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from common provider exception shapes."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return status_code

    response = getattr(exc, "response", None)
    if response is None:
        response = getattr(exc, "raw_response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and not isinstance(response_status, bool):
        return response_status

    if isinstance(response, Mapping):
        metadata = response.get("ResponseMetadata", {})
        if isinstance(metadata, Mapping):
            metadata_status = metadata.get("HTTPStatusCode")
            if isinstance(metadata_status, int) and not isinstance(metadata_status, bool):
                return metadata_status

    code = getattr(exc, "code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    return None


def _parse_retry_after(value: object) -> float | None:
    """Parse Retry-After seconds or an HTTP date into a non-negative duration."""
    if not isinstance(value, str | int | float) or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        seconds = retry_at.timestamp() - time.time()
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _model_retry_after(exc: BaseException) -> float | None:
    """Extract and parse Retry-After from common provider header containers."""
    direct_value = getattr(exc, "retry_after", None)
    if direct_value is not None:
        parsed_value = _parse_retry_after(direct_value)
        if parsed_value is not None:
            return parsed_value

    header_sets: list[Mapping] = []
    headers = getattr(exc, "headers", None)
    if isinstance(headers, Mapping):
        header_sets.append(headers)

    response = getattr(exc, "response", None)
    if response is None:
        response = getattr(exc, "raw_response", None)
    response_headers = getattr(response, "headers", None)
    if isinstance(response_headers, Mapping):
        header_sets.append(response_headers)
    if isinstance(response, Mapping):
        metadata = response.get("ResponseMetadata", {})
        if isinstance(metadata, Mapping):
            metadata_headers = metadata.get("HTTPHeaders", {})
            if isinstance(metadata_headers, Mapping):
                header_sets.append(metadata_headers)

    for header_set in header_sets:
        for key, value in header_set.items():
            if str(key).lower() == "retry-after":
                return _parse_retry_after(value)
    return None


def _model_error_name(exc: BaseException) -> str:
    """Normalize provider class names and structured error codes for policy checks."""
    names = "".join(error_type.__name__ for error_type in type(exc).__mro__).lower()
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error", {})
        if isinstance(error, Mapping):
            names += str(error.get("Code", "")).lower()
    return names.replace("_", "")


def _is_transient_model_exception(exc: BaseException, status_code: int | None) -> bool:
    """Classify only known transient provider failures as retryable."""
    if status_code is not None:
        return status_code in _TRANSIENT_HTTP_STATUSES or 500 <= status_code <= 599

    code = getattr(exc, "code", None)
    if callable(code):
        grpc_name = getattr(code(), "name", "").lower().replace("_", "")
        if grpc_name in {
            "deadlineexceeded",
            "resourceexhausted",
            "aborted",
            "unavailable",
            "internal",
        }:
            return True
        if grpc_name:
            return False

    error_name = _model_error_name(exc)
    if any(name in error_name for name in _PERMANENT_ERROR_NAMES):
        return False
    return any(name in error_name for name in _TRANSIENT_ERROR_NAMES)


def _map_exception(exc: BaseException) -> ExtractionError:
    """Translate a low-level exception into the appropriate ExtractionError subclass."""
    if isinstance(exc, httpx.HTTPStatusError):
        return UrlFetchError(f"Failed to fetch URL: {exc.response.status_code}")
    if isinstance(exc, httpx.RequestError):
        return UrlFetchError(f"Failed to fetch URL: {exc}")
    if isinstance(exc, ValidationError):
        return SchemaValidationError(f"Model output did not match schema: {exc}")
    if _is_model_exception(exc):
        status_code = _model_status_code(exc)
        return ModelError(
            f"Model API error: {exc}",
            provider=_model_provider(exc),
            status_code=status_code,
            retryable=_is_transient_model_exception(exc, status_code),
            retry_after=_model_retry_after(exc),
        )
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
    max_input_bytes: int,
) -> tuple[Agent, list]:
    """Resolve the media payload and build the agent + run inputs."""
    file_bytes, file_type = _get_media(
        input_file,
        media_type=media_type,
        max_input_bytes=max_input_bytes,
    )
    agent = _build_agent(schema, model, instructions)
    return agent, _build_run_inputs(file_bytes, file_type)


async def _prepare_run_async(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[Agent, list]:
    """Async media preparation plus agent construction."""
    file_bytes, file_type = await _get_media_async(
        input_file,
        client,
        media_type=media_type,
        max_input_bytes=max_input_bytes,
    )
    agent = _build_agent(schema, model, instructions)
    return agent, _build_run_inputs(file_bytes, file_type)


def _prepare_extraction(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int,
) -> tuple[Agent, list]:
    """Prepare one extraction while applying the public exception mapping."""
    with _extraction_errors():
        return _prepare_run(
            schema,
            model,
            input_file,
            instructions,
            media_type,
            max_input_bytes,
        )


async def _prepare_extraction_async(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[Agent, list]:
    """Prepare one async extraction while applying public exception mapping."""
    with _extraction_errors():
        return await _prepare_run_async(
            schema,
            model,
            input_file,
            instructions,
            media_type,
            max_input_bytes,
            client,
        )


async def _prepare_run_inputs_async(
    input_file: str | bytes | BinaryIO,
    media_type: str | None,
    client: httpx.AsyncClient | None = None,
    *,
    max_input_bytes: int,
) -> list:
    """Resolve one async media input and build a prompt for retry reuse."""
    with _extraction_errors():
        file_bytes, file_type = await _get_media_async(
            input_file,
            client,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
        )
        return _build_run_inputs(file_bytes, file_type)


def _run_extraction(
    agent: Agent,
    inputs: list,
):
    """Run a prepared sync extraction and return the raw pydantic-ai result."""
    with _extraction_errors():
        return agent.run_sync(inputs)


def _extract_once(
    agent: Agent,
    inputs: list,
) -> T:
    """Perform a single sync extraction attempt; return the schema instance."""
    result = _run_extraction(agent, inputs)
    return cast(T, result.output)


async def _run_extraction_async(agent: Agent, inputs: list):
    """Run a prepared async extraction with public exception mapping."""
    with _extraction_errors():
        return await agent.run(inputs)


def _retry_delay(
    retry_backoff: float,
    retry_max_backoff: float,
    attempt: int,
    retry_after: float | None,
) -> float:
    """Return bounded exponential backoff with up to 25% additive jitter."""
    if retry_after is not None:
        return min(retry_after, retry_max_backoff)
    try:
        delay = retry_backoff * (2**attempt) * (1 + random.uniform(0, 0.25))
    except OverflowError:
        return retry_max_backoff
    return min(delay, retry_max_backoff)


def _validate_retry_options(
    max_retries: object,
    retry_backoff: object,
    retry_max_backoff: object,
) -> None:
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")
    if (
        isinstance(retry_backoff, bool)
        or not isinstance(retry_backoff, int | float)
        or not math.isfinite(retry_backoff)
        or retry_backoff < 0
    ):
        raise ValueError("retry_backoff must be a finite non-negative number of seconds.")
    if (
        isinstance(retry_max_backoff, bool)
        or not isinstance(retry_max_backoff, int | float)
        or not math.isfinite(retry_max_backoff)
        or retry_max_backoff < 0
    ):
        raise ValueError("retry_max_backoff must be a finite non-negative number of seconds.")


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
    retry_max_backoff: float,
) -> R:
    """Run ``fn`` until it succeeds or transient retries are exhausted."""
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    attempt = 0
    while True:
        try:
            return fn()
        except ModelError as exc:
            if not exc.retryable or attempt >= max_retries:
                raise
            time.sleep(_retry_delay(retry_backoff, retry_max_backoff, attempt, exc.retry_after))
            attempt += 1


async def _run_with_retries_async[R](
    fn: Callable[[], Awaitable[R]],
    *,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
) -> R:
    """Async counterpart to :func:`_run_with_retries_sync`."""
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    attempt = 0
    while True:
        try:
            return await fn()
        except ModelError as exc:
            if not exc.retryable or attempt >= max_retries:
                raise
            await asyncio.sleep(
                _retry_delay(retry_backoff, retry_max_backoff, attempt, exc.retry_after)
            )
            attempt += 1


def extract(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
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
        max_input_bytes: Maximum bytes to load for this input. ``None`` uses
            ``OPENEXTRACT_MAX_INPUT_BYTES`` or the 50 MiB default.
        max_retries: Number of additional attempts after a transient
            ``ModelError``. Defaults to 0 (no retries, single attempt).
        retry_backoff: Base backoff in seconds. Sleep between attempts is
            ``retry_backoff * (2 ** attempt) * (1 + random.uniform(0, 0.25))``,
            i.e. exponential backoff with up to 25% jitter.
        retry_max_backoff: Maximum delay in seconds for exponential backoff or
            a provider ``Retry-After`` value. Defaults to 60 seconds.

    Returns:
        An instance of the schema populated with extracted data.

    Raises:
        TypeError: If ``input_file`` is bytes or file-like and ``media_type``
            is not provided.
        InputTooLargeError: If the resolved input exceeds ``max_input_bytes``.
        UrlFetchError: If the URL cannot be fetched or returns a non-2xx status.
        SchemaValidationError: If the model output doesn't match the schema.
        ModelError: If retries (if any) are exhausted.
        ExtractionError: For other extraction failures.
    """
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    limit = _resolve_max_input_bytes(max_input_bytes)
    agent, inputs = _prepare_extraction(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
    )
    return _run_with_retries_sync(
        lambda: _extract_once(agent, inputs),
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_max_backoff=retry_max_backoff,
    )


def extract_with_usage(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
) -> tuple[T, Usage]:
    """Extract structured data and return ``(output, Usage)`` for token accounting.

    Same retry semantics as :func:`extract`. Returns a :class:`Usage` describing
    the tokens consumed by the successful model call.
    """
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    limit = _resolve_max_input_bytes(max_input_bytes)
    agent, inputs = _prepare_extraction(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
    )

    def _once() -> tuple[T, Usage]:
        result = _run_extraction(agent, inputs)
        return cast(T, result.output), _usage_from_result(result)

    return _run_with_retries_sync(
        _once,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_max_backoff=retry_max_backoff,
    )


async def extract_with_usage_async(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
) -> tuple[T, Usage]:
    """Async sibling of :func:`extract_with_usage`; returns ``(output, Usage)``."""
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    limit = _resolve_max_input_bytes(max_input_bytes)
    agent, inputs = await _prepare_extraction_async(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
    )

    async def _once() -> tuple[T, Usage]:
        result = await _run_extraction_async(agent, inputs)
        return cast(T, result.output), _usage_from_result(result)

    return await _run_with_retries_async(
        _once,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_max_backoff=retry_max_backoff,
    )


async def extract_async(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
) -> T:
    """Async sibling of :func:`extract`; uses ``Agent.run`` instead of ``run_sync``."""
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    limit = _resolve_max_input_bytes(max_input_bytes)
    agent, inputs = await _prepare_extraction_async(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
    )

    async def _once() -> T:
        result = await _run_extraction_async(agent, inputs)
        return cast(T, result.output)

    return await _run_with_retries_async(
        _once,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_max_backoff=retry_max_backoff,
    )


async def _run_with_shared_agent(
    agent: Agent,
    inputs: list,
) -> object:
    """Run prepared inputs through a pre-built shared ``Agent``.

    Mirrors ``extract_async``'s error mapping so callers get the same
    ``ExtractionError`` subclasses as the per-item path.
    """
    result = await _run_extraction_async(agent, inputs)
    return result.output


async def _cancel_tasks(tasks: Iterable[asyncio.Task[object]]) -> None:
    """Cancel and await every task so no batch work outlives its caller."""
    task_list = list(tasks)
    for task in task_list:
        task.cancel()
    if task_list:
        await asyncio.gather(*task_list, return_exceptions=True)


async def _iter_extractions(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None,
    max_concurrency: int,
    return_exceptions: bool,
    media_type: str | None,
    max_input_bytes: int,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
) -> AsyncIterator[tuple[int, T | Exception]]:
    """Yield indexed batch results in completion order with bounded work."""
    file_iterator = iter(input_files)
    try:
        first_item = next(file_iterator)
    except StopIteration:
        return

    # Building the Agent (and its provider HTTP client) is ~32 ms; sharing one
    # across the batch saves ~32 ms × (N-1) per call. The Agent is stateless
    # between runs and stays inside this event loop, so this is safe.
    agent = _build_agent(schema, model, instructions)
    stop = asyncio.Event()
    pending: dict[asyncio.Task[object], int] = {}
    next_index = 0
    exhausted = False

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=_url_fetch_timeout(),
    ) as client:

        async def _run_item(item: str | bytes | BinaryIO) -> object:
            try:
                inputs = await _prepare_run_inputs_async(
                    item,
                    media_type,
                    client,
                    max_input_bytes=max_input_bytes,
                )

                async def _once():
                    # A sibling may have failed while this item was being prepared
                    # or waiting to retry. Do not begin another model call afterward.
                    if stop.is_set():
                        raise asyncio.CancelledError
                    return await _run_with_shared_agent(agent, inputs)

                return await _run_with_retries_async(
                    _once,
                    max_retries=max_retries,
                    retry_backoff=retry_backoff,
                    retry_max_backoff=retry_max_backoff,
                )
            except Exception:
                if not return_exceptions:
                    stop.set()
                raise

        def _schedule(item: str | bytes | BinaryIO) -> None:
            nonlocal next_index
            task = asyncio.create_task(_run_item(item))
            pending[task] = next_index
            next_index += 1

        def _fill_slots() -> None:
            nonlocal exhausted
            while len(pending) < max_concurrency and not exhausted:
                if next_index == 0:
                    item = first_item
                else:
                    try:
                        item = next(file_iterator)
                    except StopIteration:
                        exhausted = True
                        break
                _schedule(item)

        try:
            _fill_slots()
            while pending:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                completed: list[tuple[int, T | Exception]] = []
                failures: list[Exception] = []
                child_cancelled = False

                # Stable index ordering makes simultaneous completions deterministic.
                for task in sorted(done, key=pending.__getitem__):
                    index = pending.pop(task)
                    if task.cancelled():
                        child_cancelled = True
                        continue
                    try:
                        result = task.result()
                    except Exception as exc:
                        if return_exceptions:
                            completed.append((index, exc))
                        else:
                            failures.append(exc)
                    else:
                        completed.append((index, cast(T, result)))

                if failures:
                    await _cancel_tasks(pending)
                    pending.clear()
                    raise failures[0]
                if child_cancelled:
                    raise asyncio.CancelledError

                # Refill only after every completion has been checked for a
                # fail-fast error. Pending tasks therefore stay O(concurrency).
                _fill_slots()
                for indexed_result in completed:
                    yield indexed_result
        finally:
            await _cancel_tasks(pending)
            pending.clear()


async def _gather_extractions(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None,
    max_concurrency: int,
    return_exceptions: bool,
    media_type: str | None,
    max_input_bytes: int | None,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
) -> list:
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    _validate_max_concurrency(max_concurrency)
    limit = _resolve_max_input_bytes(max_input_bytes)
    indexed_results = [
        item
        async for item in _iter_extractions(
            schema,
            model,
            input_files,
            instructions,
            max_concurrency,
            return_exceptions,
            media_type,
            limit,
            max_retries,
            retry_backoff,
            retry_max_backoff,
        )
    ]
    indexed_results.sort(key=lambda item: item[0])
    return [result for _, result in indexed_results]


def extract_many(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
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
        max_input_bytes: Per-item byte limit. ``None`` uses
            ``OPENEXTRACT_MAX_INPUT_BYTES`` or the 50 MiB default.
        max_concurrency: Maximum number of in-flight extractions.
        return_exceptions: If True, exceptions are returned in-place instead of raised
            (mirrors :func:`asyncio.gather`).
        max_retries: Per-item retries after ``ModelError`` (same semantics as
            :func:`extract`).
        retry_backoff: Base backoff seconds between per-item retries.
        retry_max_backoff: Maximum per-item retry delay in seconds.

    Returns:
        A list of results (or exceptions, when ``return_exceptions=True``) in input order.

    Raises:
        ValueError: If ``max_concurrency`` is less than 1, ``max_retries`` is
            negative, or a backoff value is negative or non-finite.
        RuntimeError: If called from a running event loop. Use
            :func:`extract_many_async` in async code instead.
    """
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
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
            max_input_bytes,
            max_retries,
            retry_backoff,
            retry_max_backoff,
        )
    )


async def extract_many_async(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
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
        max_input_bytes,
        max_retries,
        retry_backoff,
        retry_max_backoff,
    )


def iter_extract_many_async(
    schema: type[T],
    model: str,
    input_files: Iterable[str | bytes | BinaryIO],
    instructions: str | None = None,
    *,
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_concurrency: int = 5,
    return_exceptions: bool = False,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
) -> AsyncIterator[tuple[int, T | Exception]]:
    """Stream ``(input_index, result)`` pairs in completion order.

    Unlike :func:`extract_many_async`, this API does not eagerly consume
    ``input_files`` and does not wait for the complete batch before yielding.
    At most ``max_concurrency`` items are scheduled at once. If
    ``return_exceptions`` is true, item failures are yielded as the result;
    otherwise the first failure cancels and awaits all outstanding work.
    ``max_input_bytes`` applies the same per-item cap as the list APIs.

    The function itself is synchronous because it returns an async iterator::

        async for index, result in iter_extract_many_async(...):
            ...
    """
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    _validate_max_concurrency(max_concurrency)
    limit = _resolve_max_input_bytes(max_input_bytes)
    return _iter_extractions(
        schema,
        model,
        input_files,
        instructions,
        max_concurrency,
        return_exceptions,
        media_type,
        limit,
        max_retries,
        retry_backoff,
        retry_max_backoff,
    )
