"""One-shot extraction APIs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

from ._agent import (
    _build_agent,
    _resolve_run_inputs,
    _run_extraction,
    _run_extraction_async,
    _usage_from_result,
)
from ._config import (
    _DEFAULT_RETRY_MAX_BACKOFF,
    _resolve_max_input_bytes,
    _validate_retry_options,
)
from ._errors import _extraction_errors
from ._media import _get_media, _get_media_async
from ._retry import _run_with_retries_async, _run_with_retries_sync
from ._styles import ExtractionStyle, normalize_style, prepared_style_run
from ._types import ExtractionInputLike, T, Usage

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models import Model


@contextmanager
def _agent_with_inputs(
    schema: type[BaseModel],
    model: str | Model,
    instructions: str | None,
    style: ExtractionStyle,
    file_bytes: bytes,
    file_type: str,
) -> Iterator[tuple[PydanticAgent, list]]:
    """Yield the agent and run inputs for already-resolved media.

    Shared by the sync and async preparation paths, which differ only in how
    they read the media.
    """
    with prepared_style_run(style, file_bytes, file_type) as (capabilities, style_inputs):
        with _extraction_errors():
            agent = _build_agent(
                schema,
                model,
                instructions,
                extra_capabilities=capabilities,
            )
        yield agent, _resolve_run_inputs(file_bytes, file_type, style_inputs)


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
    with _agent_with_inputs(schema, model, instructions, style, file_bytes, file_type) as prepared:
        yield prepared


@asynccontextmanager
async def _prepare_extraction_async(
    schema: type[BaseModel],
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None,
    media_type: str | None,
    max_input_bytes: int,
    style: ExtractionStyle,
) -> AsyncIterator[tuple[PydanticAgent, list]]:
    """Prepare one async extraction while applying public exception mapping."""
    with _extraction_errors():
        file_bytes, file_type = await _get_media_async(
            input_file,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
        )
    with _agent_with_inputs(schema, model, instructions, style, file_bytes, file_type) as prepared:
        yield prepared


def _extract_once(agent: PydanticAgent, inputs: list) -> BaseModel:
    """Perform a single sync extraction attempt; return the schema instance."""
    return cast(BaseModel, _run_extraction(agent, inputs).output)


def _extract_once_with_usage(agent: PydanticAgent, inputs: list) -> tuple[BaseModel, Usage]:
    """Perform a single sync attempt; return the schema instance and its usage."""
    result = _run_extraction(agent, inputs)
    return cast(BaseModel, result.output), _usage_from_result(result)


async def _extract_once_async(agent: PydanticAgent, inputs: list) -> BaseModel:
    """Perform a single async extraction attempt; return the schema instance."""
    result = await _run_extraction_async(agent, inputs)
    return cast(BaseModel, result.output)


async def _extract_once_with_usage_async(
    agent: PydanticAgent,
    inputs: list,
) -> tuple[BaseModel, Usage]:
    """Perform a single async attempt; return the schema instance and its usage."""
    result = await _run_extraction_async(agent, inputs)
    return cast(BaseModel, result.output), _usage_from_result(result)


def _oneshot_options(
    style: ExtractionStyle | str,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
    max_input_bytes: int | None,
) -> tuple[ExtractionStyle, int]:
    """Validate the retry knobs and resolve the style and byte limit up front."""
    _validate_retry_options(max_retries, retry_backoff, retry_max_backoff)
    return normalize_style(style), _resolve_max_input_bytes(max_input_bytes)


def _run_oneshot[R](
    attempt: Callable[[PydanticAgent, list], R],
    schema: type[BaseModel],
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None,
    *,
    style: ExtractionStyle | str,
    media_type: str | None,
    max_input_bytes: int | None,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
) -> R:
    """Prepare one sync extraction and run ``attempt`` under the retry policy."""
    resolved_style, limit = _oneshot_options(
        style, max_retries, retry_backoff, retry_max_backoff, max_input_bytes
    )
    with _prepare_extraction(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
        resolved_style,
    ) as (agent, inputs):
        return _run_with_retries_sync(
            lambda: attempt(agent, inputs),
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        )


async def _run_oneshot_async[R](
    attempt: Callable[[PydanticAgent, list], Awaitable[R]],
    schema: type[BaseModel],
    model: str | Model,
    input_file: ExtractionInputLike,
    instructions: str | None,
    *,
    style: ExtractionStyle | str,
    media_type: str | None,
    max_input_bytes: int | None,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
) -> R:
    """Async counterpart to :func:`_run_oneshot`."""
    resolved_style, limit = _oneshot_options(
        style, max_retries, retry_backoff, retry_max_backoff, max_input_bytes
    )
    async with _prepare_extraction_async(
        schema,
        model,
        input_file,
        instructions,
        media_type,
        limit,
        resolved_style,
    ) as (agent, inputs):
        return await _run_with_retries_async(
            lambda: attempt(agent, inputs),
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
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
    return cast(
        T,
        _run_oneshot(
            _extract_once,
            schema,
            model,
            input_file,
            instructions,
            style=style,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        ),
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

    Takes the same arguments and has the same retry semantics as
    :func:`extract`. Returns a :class:`Usage` describing the tokens consumed by
    the successful model call.
    """
    return cast(
        "tuple[T, Usage]",
        _run_oneshot(
            _extract_once_with_usage,
            schema,
            model,
            input_file,
            instructions,
            style=style,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        ),
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
    return cast(
        T,
        await _run_oneshot_async(
            _extract_once_async,
            schema,
            model,
            input_file,
            instructions,
            style=style,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        ),
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
    return cast(
        "tuple[T, Usage]",
        await _run_oneshot_async(
            _extract_once_with_usage_async,
            schema,
            model,
            input_file,
            instructions,
            style=style,
            media_type=media_type,
            max_input_bytes=max_input_bytes,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        ),
    )


__all__ = [
    "extract",
    "extract_async",
    "extract_with_usage",
    "extract_with_usage_async",
]
