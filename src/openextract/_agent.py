"""Pydantic AI agent construction and one-shot run helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar, cast

from pydantic import BaseModel

from ._config import _validate_timeout
from ._errors import _extraction_errors
from ._types import Usage
from .exceptions import ProviderNotInstalledError

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models import Model
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.settings import ModelSettings

    def Agent(*args: object, **kwargs: object) -> PydanticAgent: ...
else:

    def Agent(*args, **kwargs):
        """Construct a Pydantic AI agent without loading it at package import time."""
        from pydantic_ai import Agent as PydanticAgent

        return PydanticAgent(*args, **kwargs)


T = TypeVar("T", bound=BaseModel)

_OPENAI_PREFIX = "openai:"
_OPENAI_RESPONSES_PREFIX = "openai-responses:"

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


def _instrumentation_capabilities(
    instrument: bool | InstrumentationSettings,
) -> tuple[list[object], dict[str, object]]:
    """Return version-compatible Pydantic AI instrumentation arguments."""
    if instrument is False:
        return [], {}

    from pydantic_ai.models.instrumented import InstrumentationSettings

    if instrument is True:
        settings = InstrumentationSettings()
    elif isinstance(instrument, InstrumentationSettings):
        settings = instrument
    else:
        raise TypeError("instrument must be a bool or InstrumentationSettings instance.")

    try:
        from pydantic_ai.capabilities import Instrumentation
    except ImportError:  # pragma: no cover - compatibility with older pydantic-ai
        return [], {"instrument": settings}
    return [Instrumentation(settings)], {}


def _build_agent(
    schema: type[T],
    model: str | Model,
    instructions: str | None,
    *,
    model_settings: ModelSettings | None = None,
    instrument: bool | InstrumentationSettings = False,
    extra_capabilities: Sequence[object] = (),
) -> PydanticAgent:
    """Construct the pydantic_ai Agent, handling the ollama output-type quirk.

    A missing provider SDK surfaces here as ``ImportError`` (pydantic-ai infers
    the model eagerly); translate it into an actionable
    :class:`ProviderNotInstalledError`.
    """
    try:
        output_type = schema
        if isinstance(model, str) and model.startswith("ollama"):
            from pydantic_ai.output import NativeOutput

            output_type = NativeOutput(schema)
        capabilities, compatibility_kwargs = _instrumentation_capabilities(instrument)
        capabilities = [*capabilities, *extra_capabilities]
        routed_model = _route_model(model) if isinstance(model, str) else model
        agent_kwargs: dict[str, object] = {
            "instructions": instructions,
            "output_type": output_type,
            "model_settings": model_settings,
            **compatibility_kwargs,
        }
        if capabilities:
            agent_kwargs["capabilities"] = capabilities
        return Agent(
            routed_model,
            **agent_kwargs,
        )
    except ImportError as exc:
        if isinstance(model, str):
            message = (
                f"Model {model!r} needs a provider SDK that is not installed. "
                f"Install it with: {_install_hint(model)} "
                f"(or 'pip install openextract[all]'). Original error: {exc}"
            )
        else:
            message = f"The configured model needs a provider SDK that is not installed: {exc}"
        raise ProviderNotInstalledError(message) from exc


def _build_run_inputs(file_bytes: bytes, file_type: str) -> list:
    """Build the prompt inputs passed to the agent run."""
    from pydantic_ai import BinaryContent

    return [
        "Extract the requested information from this document.",
        BinaryContent(data=file_bytes, media_type=file_type),
    ]


def _resolve_run_inputs(
    file_bytes: bytes,
    file_type: str,
    style_inputs: list[str] | None,
) -> list:
    """Use style-specific prompts when present; otherwise pass media directly."""
    if style_inputs is None:
        return _build_run_inputs(file_bytes, file_type)
    return style_inputs


def _usage_from_result(result) -> Usage:
    """Build a ``Usage`` from a pydantic-ai run result."""
    usage_descriptor = getattr(type(result), "usage", None)
    raw = (
        result.usage
        if usage_descriptor is not None and not callable(usage_descriptor)
        else result.usage()
    )
    return Usage(
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
    )


def _model_identifier(model: str | Model, agent: object) -> str | None:
    """Return a stable model identifier for result diagnostics, when known."""
    if isinstance(model, str):
        return _route_model(model)
    name = getattr(model, "model_name", None)
    if isinstance(name, str):
        return name
    agent_model = getattr(agent, "model", None)
    name = getattr(agent_model, "model_name", None)
    return name if isinstance(name, str) else None


def _run_extraction(agent: PydanticAgent, inputs: list):
    """Run a prepared sync extraction and return the raw pydantic-ai result."""
    with _extraction_errors():
        return agent.run_sync(inputs)


async def _run_extraction_async(agent: PydanticAgent, inputs: list):
    """Run a prepared async extraction with public exception mapping."""
    with _extraction_errors():
        return await agent.run(inputs)


def _session_model_settings(
    model_settings: ModelSettings | None,
    timeout: float | None,
) -> ModelSettings | None:
    settings = dict(model_settings) if model_settings is not None else {}
    if timeout is not None:
        settings["timeout"] = _validate_timeout(timeout, name="timeout")
    return cast("ModelSettings | None", settings or None)
