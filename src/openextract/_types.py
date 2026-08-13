"""Public input, result, usage, and retry contracts."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import BinaryIO, TypeVar

from pydantic import BaseModel

from ._config import _DEFAULT_RETRY_MAX_BACKOFF, _validate_retry_options

T = TypeVar("T", bound=BaseModel)

# A raw media source accepted directly by the public APIs: a local path or
# http(s) URL string, an ``os.PathLike`` (e.g. ``pathlib.Path``), raw bytes, or
# a binary file-like object with a ``.read()`` method.
MediaSource = str | os.PathLike[str] | bytes | BinaryIO

# A media source after ``os.fspath`` normalization: no ``os.PathLike`` remains.
ResolvedSource = str | bytes | BinaryIO


@dataclass(frozen=True)
class ExtractionInput:
    """A single input for extraction with optional per-item media metadata.

    Wraps a raw :data:`MediaSource` so heterogeneous batch inputs can specify
    their own ``media_type`` (and an optional safe ``name`` for diagnostics)
    without falling back to a single batch-wide media type.

    Attributes:
        source: The media source — a local path, ``http(s)://`` URL,
            ``os.PathLike``, raw ``bytes``, or a binary file-like object.
        media_type: Optional MIME type for this item. Required when ``source``
            is ``bytes`` or a file-like object and no batch-wide ``media_type``
            override is supplied. Overrides inference for path/URL sources.
        name: Optional safe source name recorded on :class:`ExtractionResult`
            diagnostics. Never populated with raw content or credentials.
    """

    source: MediaSource
    media_type: str | None = None
    name: str | None = None


# Anything accepted as a single input or batch item: a raw :data:`MediaSource`
# or a structured :class:`ExtractionInput`.
ExtractionInputLike = MediaSource | ExtractionInput


@dataclass(frozen=True)
class Usage:
    """Token usage information for a single extraction call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ExtractionResult[T]:
    """Diagnostics-rich result of one extraction.

    ``extract_many_with_results`` returns these so callers can account for
    token usage, observe retries and timing, and record safe provenance without
    retaining raw media, credentials, query strings, or provider internals.

    Attributes:
        output: The validated schema instance.
        usage: Token usage from the successful model call.
        attempts: Number of model-call attempts, including the initial call and
            any :class:`ModelError` retries (always ``>= 1`` on success).
        duration: Wall-clock seconds spent on this item, including retries.
        model: The model identifier that produced the output, when known.
        media_type: The media type requested for this item, when provided.
        source: A sanitized source label (``ExtractionInput.name``, or a
            credential/query-stripped path/URL context); ``None`` for unnamed
            bytes/file-like inputs.
        warnings: Extensible, currently empty diagnostics channel.
    """

    output: T
    usage: Usage
    attempts: int
    duration: float
    model: str | None
    media_type: str | None
    source: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetryPolicy:
    """Retry configuration shared by every call made through an extractor session."""

    max_retries: int = 0
    backoff: float = 1.0
    max_backoff: float = _DEFAULT_RETRY_MAX_BACKOFF

    def __post_init__(self) -> None:
        _validate_retry_options(self.max_retries, self.backoff, self.max_backoff)


def total_usage(results: Iterable[ExtractionResult[T]]) -> Usage:
    """Sum token usage across batch extraction results.

    ``results`` typically comes from :func:`extract_many_with_results` or
    :func:`extract_many_with_results_async`. Only successful items carry a
    :class:`Usage`, so totals reflect the successful calls in the batch.
    """
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for result in results:
        input_tokens += result.usage.input_tokens
        output_tokens += result.usage.output_tokens
        total_tokens += result.usage.total_tokens
    return Usage(input_tokens, output_tokens, total_tokens)


def _resolve_item(
    item: ExtractionInputLike,
    global_media_type: str | None,
) -> tuple[MediaSource, str | None, str | None]:
    """Split a batch item into ``(source, effective media type, safe name)``.

    A per-item :class:`ExtractionInput` media type wins over the batch-wide
    ``media_type`` fallback so heterogeneous inputs can use different types.
    """
    if isinstance(item, ExtractionInput):
        media_type = item.media_type if item.media_type is not None else global_media_type
        return item.source, media_type, item.name
    return item, global_media_type, None
