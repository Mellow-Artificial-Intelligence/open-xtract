"""Importable extract agents: local definitions, remote endpoints, discovery.

An agent packages the choices a swarm member would otherwise repeat at every
call site — model, style, instructions, and the output schema — so a caller can
import a specialist instead of re-specifying it. Agents compose: a parent with
``subagents`` flattens into one swarm member per leaf.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

from pydantic import BaseModel

from ._styles import ExtractionStyle

if TYPE_CHECKING:
    from pydantic_ai.models import Model

_AGENT_FILE = "agent.py"
_SUBAGENTS_DIR = "subagents"
_INSTRUCTIONS_FILE = "instructions.md"
_module_counter = itertools.count()

# A header provider: a mapping, or a callable returning one (optionally async).
type HeadersInput = dict[str, str] | Callable[[], dict[str, str] | Awaitable[dict[str, str]]] | None
# An outbound auth provider, as built by the helpers in ``openextract.auth``.
type AuthFn = Callable[[], dict[str, str] | Awaitable[dict[str, str]]]
# A URL that may be resolved lazily, e.g. from a rotating deployment record.
type UrlInput = str | Callable[[], str | Awaitable[str]]


@dataclass(frozen=True)
class RemoteAgent:
    """An extraction agent reachable over HTTP instead of a local model.

    The endpoint receives the JSON Schema, the base64 media, and the style, and
    answers with ``{"output": ...}``. ``url`` may be a callable so a rotating
    deployment URL is resolved per call, and ``auth`` supplies request headers.
    """

    url: UrlInput
    description: str
    auth: AuthFn | None = None
    headers: HeadersInput = None
    path: str = "/extract"
    output_schema: type[BaseModel] | None = None


@dataclass(frozen=True)
class DefinedAgent:
    """A local extraction agent: a model plus its standing configuration.

    An agent with ``subagents`` and no ``model`` is a pure group; one with both
    contributes itself and its children.
    """

    description: str
    model: str | Model | None = None
    style: ExtractionStyle | str | None = None
    instructions: str | None = None
    output_schema: type[BaseModel] | None = None
    subagents: tuple[AgentInput, ...] = ()


@dataclass(frozen=True)
class SwarmMember:
    """One agent in a swarm: a model or remote endpoint plus overrides.

    ``instructions`` and ``style`` win over the swarm-wide values, so a single
    swarm can pair a ``search`` reader with a ``direct`` reader over the same
    document.
    """

    model: str | Model | RemoteAgent
    instructions: str | None = None
    style: ExtractionStyle | str | None = None


# Anything usable as a swarm agent: a model identifier, a configured
# pydantic-ai ``Model``, a :class:`SwarmMember`, or a defined/remote agent.
type AgentInput = str | Model | SwarmMember | DefinedAgent | RemoteAgent


async def resolve_provided(value: Any) -> Any:
    """Resolve an agent config value that may be a sync or async provider.

    Lazy URLs, headers, and auth are read per request so a rotated credential
    is picked up without redefining the agent.
    """
    if not callable(value):
        return value
    resolved = value()
    if isinstance(resolved, Awaitable):
        return await resolved
    return resolved


def _require_description(description: object) -> str:
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required.")
    return description.strip()


def _validate_output_schema(output_schema: object) -> type[BaseModel] | None:
    if output_schema is None:
        return None
    if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
        return output_schema
    raise TypeError("output_schema must be a Pydantic BaseModel subclass.")


def define_agent(
    description: str,
    *,
    model: str | Model | None = None,
    style: ExtractionStyle | str | None = None,
    instructions: str | None = None,
    output_schema: type[BaseModel] | None = None,
    subagents: Sequence[AgentInput] = (),
) -> DefinedAgent:
    """Define a local extraction agent.

    Args:
        description: What this agent extracts. Required; it is how a caller
            picks an agent out of a directory.
        model: The model identifier or configured pydantic-ai ``Model``.
            Optional only when ``subagents`` is non-empty.
        style: Extraction style for this agent, overriding the call site.
        instructions: Standing guidance for this agent.
        output_schema: The Pydantic model this agent produces, so callers can
            extract without naming a schema.
        subagents: Child agents that fan out alongside (or instead of) this one.

    Raises:
        ValueError: If ``description`` is blank, or neither ``model`` nor
            ``subagents`` is given.
        TypeError: If ``output_schema`` is not a ``BaseModel`` subclass.
    """
    resolved = _require_description(description)
    children = tuple(subagents)
    if model is None and not children:
        raise ValueError("define_agent requires model or subagents.")
    return DefinedAgent(
        description=resolved,
        model=model,
        style=style,
        instructions=instructions,
        output_schema=_validate_output_schema(output_schema),
        subagents=children,
    )


def define_remote_agent(
    url: UrlInput,
    description: str,
    *,
    auth: AuthFn | None = None,
    headers: HeadersInput = None,
    path: str = "/extract",
    output_schema: type[BaseModel] | None = None,
) -> RemoteAgent:
    """Define an extraction agent served over HTTP.

    Args:
        url: The agent's base URL, or a callable resolving one per call.
        description: What the remote agent extracts. Required.
        auth: Outbound auth header provider; see :mod:`openextract.auth`.
        headers: Extra request headers, or a callable returning them.
        path: Path appended to ``url``. Defaults to ``/extract``.
        output_schema: The Pydantic model the endpoint returns.

    Raises:
        ValueError: If ``description`` or ``url`` is blank.
        TypeError: If ``output_schema`` is not a ``BaseModel`` subclass.
    """
    resolved = _require_description(description)
    if not callable(url) and (not isinstance(url, str) or not url.strip()):
        raise ValueError("url is required.")
    return RemoteAgent(
        url=url,
        description=resolved,
        auth=auth,
        headers=headers,
        path=path,
        output_schema=_validate_output_schema(output_schema),
    )


def is_agent(value: object) -> TypeGuard[DefinedAgent | RemoteAgent]:
    """Return whether ``value`` came from ``define_agent`` / ``define_remote_agent``."""
    return isinstance(value, DefinedAgent | RemoteAgent)


def resolve_output_schema(agent: DefinedAgent | RemoteAgent) -> type[BaseModel]:
    """Return the agent's declared output schema.

    Raises:
        ValueError: If the agent was defined without an ``output_schema``.
    """
    if agent.output_schema is None:
        raise ValueError(f"agent {agent.description!r} is missing output_schema.")
    return agent.output_schema


def flatten_agent(agent: AgentInput) -> list[SwarmMember]:
    """Expand one agent into the swarm members it contributes, in order.

    A group agent contributes its own model (when it has one) followed by each
    subagent's members, depth first.
    """
    if isinstance(agent, DefinedAgent):
        members = (
            [SwarmMember(agent.model, agent.instructions, agent.style)]
            if agent.model is not None
            else []
        )
        for child in agent.subagents:
            members.extend(flatten_agent(child))
        return members
    if isinstance(agent, RemoteAgent):
        return [SwarmMember(agent)]
    if isinstance(agent, SwarmMember):
        return [agent]
    return [SwarmMember(agent)]


def _import_agent_module(path: Path) -> Any:
    """Import a Python file as a throwaway module so its agent can be read."""
    name = f"openextract_agent_{path.stem}_{next(_module_counter)}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise ValueError(f"Cannot import agent module '{path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _agent_from_module(module: Any, label: str) -> DefinedAgent | RemoteAgent:
    agent = getattr(module, "agent", None)
    if is_agent(agent):
        return agent
    raise ValueError(f"'{label}' must define an 'agent' from define_agent or define_remote_agent.")


def _load_module_attribute(spec: str) -> DefinedAgent | RemoteAgent:
    """Resolve a ``module:attribute`` agent reference."""
    module_part, _, attribute = spec.partition(":")
    if not module_part or not attribute:
        raise ValueError(
            f"Invalid agent path '{spec}'. Expected a directory, a Python file, "
            "or 'module:attribute'."
        )
    module = importlib.import_module(module_part)
    agent = getattr(module, attribute, None)
    if is_agent(agent):
        return agent
    raise ValueError(
        f"'{attribute}' in module '{module_part}' is not a define_agent "
        "or define_remote_agent value."
    )


def _skip_entry(name: str) -> bool:
    return name.startswith(".") or name.startswith("_") or name == _AGENT_FILE


def _load_subagents(directory: Path) -> list[DefinedAgent | RemoteAgent]:
    """Load every agent under a ``subagents/`` directory, in sorted order."""
    if not directory.is_dir():
        return []
    agents: list[DefinedAgent | RemoteAgent] = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if _skip_entry(entry.name):
            continue
        if entry.is_dir():
            agents.append(load_agent_directory(entry))
        elif entry.suffix == ".py":
            agents.append(_agent_from_module(_import_agent_module(entry), str(entry)))
    return agents


def _directory_instructions(directory: Path) -> str | None:
    """Return ``instructions.md`` content as the directory's default guidance."""
    path = directory / _INSTRUCTIONS_FILE
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def load_agent_directory(directory: str | Path) -> DefinedAgent | RemoteAgent:
    """Load an agent from a directory.

    The directory may hold ``agent.py`` (defining ``agent``), a ``subagents/``
    directory of further agents, and ``instructions.md`` used as the agent's
    instructions when it declared none. A directory with only subagents becomes
    a group agent named after the directory.

    Raises:
        ValueError: If the directory has neither ``agent.py`` nor subagents.
    """
    path = Path(directory)
    agent_file = path / _AGENT_FILE
    root = (
        _agent_from_module(_import_agent_module(agent_file), str(agent_file))
        if agent_file.is_file()
        else None
    )
    children = _load_subagents(path / _SUBAGENTS_DIR)
    if root is not None:
        if isinstance(root, RemoteAgent):
            return root
        return replace(
            root,
            instructions=root.instructions or _directory_instructions(path),
            subagents=(*root.subagents, *children),
        )
    if len(children) == 1:
        return children[0]
    if children:
        return define_agent(
            path.resolve().name,
            instructions=_directory_instructions(path),
            subagents=children,
        )
    raise ValueError(f"No {_AGENT_FILE} or {_SUBAGENTS_DIR}/ found in '{path}'.")


def load_agent(spec: str | Path | DefinedAgent | RemoteAgent) -> DefinedAgent | RemoteAgent:
    """Load one agent from a directory, a Python file, or ``module:attribute``.

    An already-defined agent is returned unchanged so callers can accept either.

    Raises:
        ValueError: If ``spec`` is not a usable reference or the target does not
            define an agent.
    """
    if is_agent(spec):
        return spec
    if isinstance(spec, Path):
        spec = str(spec)
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(
            "agent must be a define_agent value, a directory, a Python file, "
            "or a 'module:attribute' path."
        )
    trimmed = spec.strip()
    path = Path(trimmed)
    if path.is_dir():
        return load_agent_directory(path)
    if path.is_file():
        return _agent_from_module(_import_agent_module(path), trimmed)
    return _load_module_attribute(trimmed)


def load_agents(
    spec: str | Sequence[str | Path | DefinedAgent | RemoteAgent],
) -> list[DefinedAgent | RemoteAgent]:
    """Load several agents from a comma-separated string or a sequence.

    Raises:
        ValueError: If no agent reference is given.
    """
    items: Sequence[Any] = (
        [item.strip() for item in spec.split(",") if item.strip()]
        if isinstance(spec, str)
        else list(spec)
    )
    if not items:
        raise ValueError("agents must include at least one agent path.")
    return [load_agent(item) for item in items]
