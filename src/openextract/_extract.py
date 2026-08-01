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
_URL_TIMEOUT_ENV = "OPENEXTRACT_URL_TIMEOUT"
_MAX_REDIRECTS_ENV = "OPENEXTRACT_MAX_REDIRECTS"
_ALLOW_PRIVATE_URLS_ENV = "OPENEXTRACT_ALLOW_PRIVATE_URLS"
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


def _media_from_response(file_path: str, response: httpx.Response) -> tuple[bytes, str]:
    """Resolve response bytes and MIME type for a URL input."""
    media_type = _get_media_type(file_path)
    if media_type == _DEFAULT_MEDIA_TYPE:
        header = response.headers.get("content-type", "").split(";", 1)[0].strip()
        if header:
            media_type = header
    return response.content, media_type


def _read_from_path(file_path: str) -> tuple[bytes, str]:
    """Read bytes from a local path or http(s) URL; return (bytes, media_type)."""
    if file_path.startswith(_URL_PREFIXES):
        response = _fetch_url(file_path)
        return _media_from_response(file_path, response)

    return Path(file_path).read_bytes(), _get_media_type(file_path)


async def _read_from_path_async(
    file_path: str,
    client: httpx.AsyncClient | None,
) -> tuple[bytes, str]:
    """Async counterpart to :func:`_read_from_path`."""
    if file_path.startswith(_URL_PREFIXES):
        if client is None:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=_url_fetch_timeout(),
            ) as owned_client:
                response = await _fetch_url_async(file_path, owned_client)
        else:
            response = await _fetch_url_async(file_path, client)
        return _media_from_response(file_path, response)

    media_bytes = await asyncio.to_thread(Path(file_path).read_bytes)
    return media_bytes, _get_media_type(file_path)


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


async def _get_media_async(
    input_file: str | bytes | BinaryIO,
    client: httpx.AsyncClient | None = None,
    media_type: str | None = None,
) -> tuple[bytes, str]:
    """Resolve media without blocking the event loop on disk, DNS, or stream I/O."""
    if isinstance(input_file, str):
        file_bytes, resolved_type = await _read_from_path_async(input_file, client)
        return file_bytes, media_type or resolved_type

    if isinstance(input_file, bytes):
        return _get_media(input_file, media_type=media_type)

    if hasattr(input_file, "read"):
        return await asyncio.to_thread(_get_media, input_file, media_type)

    return _get_media(input_file, media_type=media_type)


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
) -> tuple[Agent, list]:
    """Load env, resolve the media payload, and build the agent + run inputs."""
    _ensure_dotenv_loaded()
    file_bytes, file_type = _get_media(input_file, media_type=media_type)
    agent = _build_agent(schema, model, instructions)
    return agent, _build_run_inputs(file_bytes, file_type)


async def _prepare_run_async(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
    client: httpx.AsyncClient | None = None,
) -> tuple[Agent, list]:
    """Async media preparation plus agent construction."""
    _ensure_dotenv_loaded()
    file_bytes, file_type = await _get_media_async(input_file, client, media_type=media_type)
    agent = _build_agent(schema, model, instructions)
    return agent, _build_run_inputs(file_bytes, file_type)


def _prepare_extraction(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
) -> tuple[Agent, list]:
    """Prepare one extraction while applying the public exception mapping."""
    with _extraction_errors():
        return _prepare_run(schema, model, input_file, instructions, media_type)


async def _prepare_extraction_async(
    schema: type[T],
    model: str,
    input_file: str | bytes | BinaryIO,
    instructions: str | None,
    media_type: str | None,
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
            client,
        )


async def _prepare_run_inputs_async(
    input_file: str | bytes | BinaryIO,
    media_type: str | None,
    client: httpx.AsyncClient | None = None,
) -> list:
    """Resolve one async media input and build a prompt for retry reuse."""
    with _extraction_errors():
        file_bytes, file_type = await _get_media_async(
            input_file,
            client,
            media_type=media_type,
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
    _validate_retry_options(max_retries, retry_backoff)
    agent, inputs = _prepare_extraction(schema, model, input_file, instructions, media_type)
    return _run_with_retries_sync(
        lambda: _extract_once(agent, inputs),
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
) -> tuple[T, Usage]:
    """Extract structured data and return ``(output, Usage)`` for token accounting.

    Same retry semantics as :func:`extract`. Returns a :class:`Usage` describing
    the tokens consumed by the successful model call.
    """
    _validate_retry_options(max_retries, retry_backoff)
    agent, inputs = _prepare_extraction(schema, model, input_file, instructions, media_type)

    def _once() -> tuple[T, Usage]:
        result = _run_extraction(agent, inputs)
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
) -> tuple[T, Usage]:
    """Async sibling of :func:`extract_with_usage`; returns ``(output, Usage)``."""
    _validate_retry_options(max_retries, retry_backoff)
    agent, inputs = await _prepare_extraction_async(
        schema,
        model,
        input_file,
        instructions,
        media_type,
    )

    async def _once() -> tuple[T, Usage]:
        result = await _run_extraction_async(agent, inputs)
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
) -> T:
    """Async sibling of :func:`extract`; uses ``Agent.run`` instead of ``run_sync``."""
    _validate_retry_options(max_retries, retry_backoff)
    agent, inputs = await _prepare_extraction_async(
        schema,
        model,
        input_file,
        instructions,
        media_type,
    )

    async def _once() -> T:
        result = await _run_extraction_async(agent, inputs)
        return cast(T, result.output)

    return await _run_with_retries_async(
        _once,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
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
) -> list:
    _validate_retry_options(max_retries, retry_backoff)
    _validate_max_concurrency(max_concurrency)
    files = list(input_files)
    if not files:
        return []
    _ensure_dotenv_loaded()
    # Building the Agent (and its provider HTTP client) is ~32 ms; sharing one
    # across the batch saves ~32 ms × (N-1) per call. The Agent is stateless
    # between runs and stays inside this event loop, so this is safe.
    agent = _build_agent(schema, model, instructions)
    semaphore = asyncio.Semaphore(max_concurrency)

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=_url_fetch_timeout(),
    ) as client:

        async def _bounded(item):
            async with semaphore:
                inputs = await _prepare_run_inputs_async(item, media_type, client)

                async def _once():
                    return await _run_with_shared_agent(agent, inputs)

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

    Returns:
        A list of results (or exceptions, when ``return_exceptions=True``) in input order.

    Raises:
        ValueError: If ``max_concurrency`` is less than 1, ``max_retries`` is
            negative, or ``retry_backoff`` is not positive and finite.
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
    )
