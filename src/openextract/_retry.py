"""Retry helpers shared by one-shot, session, and batch extraction."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

from ._config import _validate_retry_options
from .exceptions import ModelError


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
