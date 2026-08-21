"""HTTP transport for remote extraction agents."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import httpx

from ._agents import RemoteAgent, resolve_provided
from ._config import _url_fetch_timeout
from ._errors import _extraction_errors
from ._media import _require_safe_url_async
from ._retry import _run_with_retries_async
from ._styles import ExtractionStyle
from ._types import T, Usage
from .exceptions import RemoteAgentError

if TYPE_CHECKING:
    from pydantic import BaseModel

_JSON_HEADERS = {"accept": "application/json", "content-type": "application/json"}
_TRANSIENT_STATUSES = frozenset((408, 409, 425, 429))


async def _resolve_agent_url(agent: RemoteAgent) -> str:
    """Resolve, validate, and join the agent's request URL.

    The host goes through the same SSRF allowlist as document URLs, so an agent
    endpoint cannot be pointed at link-local metadata by a rotating URL
    provider. Local agent servers need ``OPENEXTRACT_ALLOW_PRIVATE_URLS=1``.
    """
    value = await resolve_provided(agent.url)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("remote agent url must resolve to a non-empty string.")
    base = value.strip()
    if urlparse(base).scheme not in ("http", "https"):
        raise ValueError(f"remote agent url must be http or https: {base}")
    await _require_safe_url_async(base)
    suffix = agent.path if agent.path.startswith("/") else f"/{agent.path}"
    return f"{base.rstrip('/')}{suffix}"


async def _resolve_headers(agent: RemoteAgent) -> dict[str, str]:
    """Merge JSON defaults, the agent's extra headers, and its auth headers."""
    extra = await resolve_provided(agent.headers) or {}
    auth = await resolve_provided(agent.auth) if agent.auth is not None else {}
    return {**_JSON_HEADERS, **extra, **auth}


def _usage_from_payload(value: Any) -> Usage:
    """Read token usage from a remote response, accepting either key style."""
    if not isinstance(value, dict):
        return Usage(0, 0, 0)

    def field(camel: str, snake: str) -> int:
        raw = value.get(camel, value.get(snake, 0))
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else 0

    return Usage(
        input_tokens=field("inputTokens", "input_tokens"),
        output_tokens=field("outputTokens", "output_tokens"),
        total_tokens=field("totalTokens", "total_tokens"),
    )


def _decode(response: httpx.Response, url: str) -> Any:
    """Parse a JSON response body, mapping every failure onto RemoteAgentError."""
    try:
        payload = response.json() if response.content else None
    except ValueError:
        raise RemoteAgentError(
            f"Remote agent returned non-JSON ({response.status_code}).",
            url=url,
            status_code=response.status_code,
        ) from None
    if not response.is_success:
        message = (
            payload["error"]
            if isinstance(payload, dict) and isinstance(payload.get("error"), str)
            else f"Remote agent failed with status {response.status_code}."
        )
        raise RemoteAgentError(message, url=url, status_code=response.status_code)
    if not isinstance(payload, dict):
        raise RemoteAgentError(
            "Remote agent returned an empty response.",
            url=url,
            status_code=response.status_code,
            retryable=False,
        )
    if isinstance(payload.get("error"), str):
        raise RemoteAgentError(
            payload["error"], url=url, status_code=response.status_code, retryable=False
        )
    return payload


async def run_remote_extraction(
    schema: type[T],
    agent: RemoteAgent,
    file_bytes: bytes,
    file_type: str,
    *,
    instructions: str | None,
    style: ExtractionStyle,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
) -> tuple[T, Usage, int]:
    """POST the media to a remote agent and validate its answer.

    Returns ``(output, usage, attempts)``. The request body matches the
    openextract agent protocol: the JSON Schema, base64 media with its
    ``mediaType``, the instructions, and the style.
    """
    url = await _resolve_agent_url(agent)
    body = {
        "schema": cast("type[BaseModel]", schema).model_json_schema(),
        "input": {
            "data": base64.b64encode(file_bytes).decode("ascii"),
            "mediaType": file_type,
        },
        "instructions": instructions,
        "style": style.value,
    }
    attempts = 0

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=_url_fetch_timeout(),
    ) as client:

        async def _once() -> tuple[T, Usage]:
            nonlocal attempts
            attempts += 1
            try:
                response = await client.post(url, json=body, headers=await _resolve_headers(agent))
            except httpx.HTTPError as exc:
                raise RemoteAgentError(
                    f"Remote agent request failed: {exc}", url=url, retryable=True
                ) from exc
            payload = _decode(response, url)
            with _extraction_errors():
                output = cast(
                    T,
                    cast("type[BaseModel]", schema).model_validate(payload.get("output", payload)),
                )
            return output, _usage_from_payload(payload.get("usage"))

        output, usage = await _run_with_retries_async(
            _once,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        )
    return output, usage, attempts


__all__ = ["run_remote_extraction"]
