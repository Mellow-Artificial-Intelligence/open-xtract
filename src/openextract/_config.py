"""Shared constants, environment knobs, and option validation."""

from __future__ import annotations

import math
import os

_DEFAULT_URL_FETCH_TIMEOUT = 30.0
_DEFAULT_MAX_REDIRECTS = 10
_DEFAULT_RETRY_MAX_BACKOFF = 60.0
_DEFAULT_MAX_INPUT_BYTES = 50 * 1024 * 1024
_MAX_SWARM_SIZE = 16
_URL_TIMEOUT_ENV = "OPENEXTRACT_URL_TIMEOUT"
_MAX_REDIRECTS_ENV = "OPENEXTRACT_MAX_REDIRECTS"
_ALLOW_PRIVATE_URLS_ENV = "OPENEXTRACT_ALLOW_PRIVATE_URLS"
_MAX_INPUT_BYTES_ENV = "OPENEXTRACT_MAX_INPUT_BYTES"


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


def _allow_private_urls() -> bool:
    """Return True when SSRF host validation is disabled via env var."""
    return os.environ.get(_ALLOW_PRIVATE_URLS_ENV, "").lower() in ("1", "true", "yes")


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


def _validate_swarm_size(size: object) -> int:
    """Validate a swarm agent count and return it.

    The upper bound keeps a typo (``size=1000``) from fanning out into a
    provider-rate-limit incident before any model call is made.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or size > _MAX_SWARM_SIZE:
        raise ValueError(f"size must be an integer from 1 to {_MAX_SWARM_SIZE}.")
    return size


def _validate_timeout(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number of seconds.")
    return float(value)
