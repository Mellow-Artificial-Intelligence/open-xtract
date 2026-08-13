"""One-shot extraction APIs and internal re-exports."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, TypeVar, cast

import httpx
from pydantic import BaseModel

from ._agent import (
    Agent,
    _build_agent,
    _install_hint,
    _model_identifier,
    _resolve_run_inputs,
    _route_model,
    _run_extraction,
    _run_extraction_async,
    _usage_from_result,
)
from ._batch import (
    _run_with_shared_agent,
    _run_with_shared_agent_result,
    extract_many,
    extract_many_async,
    extract_many_with_results,
    extract_many_with_results_async,
    iter_extract_many_async,
)
from ._config import (
    _DEFAULT_RETRY_MAX_BACKOFF,
    _max_redirects,
    _resolve_max_input_bytes,
    _url_fetch_timeout,
    _validate_retry_options,
)
from ._errors import (
    _extraction_errors,
    _is_transient_model_exception,
    _map_exception,
    _model_retry_after,
    _model_status_code,
    _parse_retry_after,
)
from ._media import (
    _fetch_url,
    _fetch_url_async,
    _get_media,
    _get_media_async,
    _get_media_type,
    _is_public_ip,
    _is_safe_host,
    _item_source_label,
    _read_from_path,
    _read_url_with_client,
    _safe_source_context,
)
from ._retry import _retry_delay, _run_with_retries_async, _run_with_retries_sync
from ._session import AsyncExtractor, Extractor
from ._styles import ExtractionStyle, normalize_style, prepared_style_run
from ._types import (
    ExtractionInput,
    ExtractionInputLike,
    ExtractionResult,
    RetryPolicy,
    Usage,
    _resolve_item,
    total_usage,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models import Model

T = TypeVar("T", bound=BaseModel)


@contextmanager
def _prepare_extraction(
    schema: type[BaseModel],
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int,
    style: ExtractionStyle,
) -> Iterator[tuple[PydanticAgent, list]]:
    """Prepare one extraction while applying the public exception mapping."""
    with _extraction_errors():
        file_bytes, file_type = _get_media(
            input_file,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
        )
    with prepared_style_run(style, file_bytes, file_type) as (capabilities, style_inputs):
        with _extraction_errors():
            agent = _build_agent(
                schema,
                model,
                instructions,
                extra_capabilities=capabilities,
            )
        yield agent, _resolve_run_inputs(file_bytes, file_type, style_inputs)


@asynccontextmanager
async def _prepare_extraction_async(
    schema: type[BaseModel],
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int,
    style: ExtractionStyle,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[tuple[PydanticAgent, list]]:
    """Prepare one async extraction while applying public exception mapping."""
    with _extraction_errors():
        file_bytes, file_type = await _get_media_async(
            input_file,
            client,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
        )
    with prepared_style_run(style, file_bytes, file_type) as (capabilities, style_inputs):
        with _extraction_errors():
            agent = _build_agent(
                schema,
                model,
                instructions,
                extra_capabilities=capabilities,
            )
        yield agent, _resolve_run_inputs(file_bytes, file_type, style_inputs)


def _extract_once(
    agent: PydanticAgent,
    inputs: list,
) -> T:
    """Perform a single sync extraction attempt; return the schema instance."""
    result = _run_extraction(agent, inputs)
    return cast(T, result.output)


def _oneshot_options(
    style: ExtractionStyle | str,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
    max_input_bytes: int | None,
) -> tuple[ExtractionStyle, int, int, float, float]:
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    return (
        normalize_style(style),
        _resolve_max_input_bytes(max_input_bytes),
        max_retries,
        retry_backoff,
        retry_max_backoff,
    )


def extract(
    schema: type[T],
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
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
        style: Extraction strategy. ``direct`` (default) sends the media to the
            model in one shot. ``search`` gives the model file tools (grep/read)
            against a text document. ``code`` lets the model write Python against
            a text document via the Pydantic AI harness.
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
        ProviderNotInstalledError: If a provider SDK or style extra is missing.
        ExtractionError: For other extraction failures.
        ValueError: If ``style`` is invalid or ``search``/``code`` is used with
            a non-text document.
    """
    style, limit, max_retries, retry_backoff, retry_max_backoff = _oneshot_options(
        style, max_retries, retry_backoff, retry_max_backoff, max_input_bytes
    )
    with _prepare_extraction(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
        style,
    ) as (agent, inputs):
        return _run_with_retries_sync(
            lambda: _extract_once(agent, inputs),
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        )


def extract_with_usage(
    schema: type[T],
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
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
    style, limit, max_retries, retry_backoff, retry_max_backoff = _oneshot_options(
        style, max_retries, retry_backoff, retry_max_backoff, max_input_bytes
    )
    with _prepare_extraction(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
        style,
    ) as (agent, inputs):

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
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
) -> tuple[T, Usage]:
    """Async sibling of :func:`extract_with_usage`; returns ``(output, Usage)``."""
    style, limit, max_retries, retry_backoff, retry_max_backoff = _oneshot_options(
        style, max_retries, retry_backoff, retry_max_backoff, max_input_bytes
    )
    async with _prepare_extraction_async(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
        style,
    ) as (agent, inputs):

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
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None = None,
    *,
    style: ExtractionStyle | str = "direct",
    media_type: str | None = None,
    max_input_bytes: int | None = None,
    max_retries: int = 0,
    retry_backoff: float = 1.0,
    retry_max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF,
) -> T:
    """Async sibling of :func:`extract`; uses ``Agent.run`` instead of ``run_sync``."""
    style, limit, max_retries, retry_backoff, retry_max_backoff = _oneshot_options(
        style, max_retries, retry_backoff, retry_max_backoff, max_input_bytes
    )
    async with _prepare_extraction_async(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
        style,
    ) as (agent, inputs):

        async def _once() -> T:
            result = await _run_extraction_async(agent, inputs)
            return cast(T, result.output)

        return await _run_with_retries_async(
            _once,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        )


__all__ = [
    "Agent",
    "Extractor",
    "AsyncExtractor",
    "RetryPolicy",
    "ExtractionInput",
    "ExtractionResult",
    "Usage",
    "extract",
    "extract_async",
    "extract_with_usage",
    "extract_with_usage_async",
    "extract_many",
    "extract_many_async",
    "iter_extract_many_async",
    "extract_many_with_results",
    "extract_many_with_results_async",
    "total_usage",
    "_build_agent",
    "_extract_once",
    "_fetch_url",
    "_fetch_url_async",
    "_get_media",
    "_get_media_async",
    "_get_media_type",
    "_install_hint",
    "_is_public_ip",
    "_is_safe_host",
    "_is_transient_model_exception",
    "_item_source_label",
    "_map_exception",
    "_max_redirects",
    "_model_identifier",
    "_model_retry_after",
    "_model_status_code",
    "_parse_retry_after",
    "_prepare_extraction",
    "_read_from_path",
    "_read_url_with_client",
    "_resolve_item",
    "_resolve_max_input_bytes",
    "_resolve_run_inputs",
    "_retry_delay",
    "_route_model",
    "_run_extraction",
    "_run_with_shared_agent",
    "_run_with_shared_agent_result",
    "_safe_source_context",
    "_url_fetch_timeout",
]
