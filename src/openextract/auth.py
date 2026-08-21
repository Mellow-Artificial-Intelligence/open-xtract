"""Outbound auth header providers for remote extraction agents.

Each helper returns a callable that :func:`openextract.define_remote_agent`
invokes per request, so a rotating token is read at call time rather than
captured once at import time.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Awaitable, Callable

from ._agents import resolve_provided

_VERCEL_OIDC_TOKEN_ENV = "VERCEL_OIDC_TOKEN"

type _Provided[V] = V | Callable[[], V | Awaitable[V]]


def bearer(token: _Provided[str]) -> Callable[[], Awaitable[dict[str, str]]]:
    """Send ``Authorization: Bearer <token>``.

    ``token`` may be a string or a callable (sync or async) resolved per request.
    """

    async def provide() -> dict[str, str]:
        return {"Authorization": f"Bearer {await resolve_provided(token)}"}

    return provide


def basic(credentials: _Provided[tuple[str, str]]) -> Callable[[], Awaitable[dict[str, str]]]:
    """Send HTTP Basic auth for a ``(username, password)`` pair.

    ``credentials`` may be the pair itself or a callable (sync or async)
    resolved per request.
    """

    async def provide() -> dict[str, str]:
        username, password = await resolve_provided(credentials)
        encoded = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        return {"Authorization": f"Basic {encoded}"}

    return provide


def vercel_oidc() -> Callable[[], Awaitable[dict[str, str]]]:
    """Send the Vercel OIDC token from ``VERCEL_OIDC_TOKEN`` as a bearer token.

    The variable is read per request so a refreshed token is picked up without
    redefining the agent.

    Raises:
        ValueError: If ``VERCEL_OIDC_TOKEN`` is unset or empty.
    """

    async def provide() -> dict[str, str]:
        token = os.environ.get(_VERCEL_OIDC_TOKEN_ENV, "").strip()
        if not token:
            raise ValueError(f"{_VERCEL_OIDC_TOKEN_ENV} is not set.")
        return {"Authorization": f"Bearer {token}"}

    return provide
