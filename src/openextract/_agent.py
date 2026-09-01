"""Pydantic AI agent construction and one-shot run helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from ._config import _validate_timeout
from ._errors import _extraction_errors
from ._types import T, Usage
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
            "model_settings": _usage_model_settings(model, model_settings),
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


_INPUT_TOKEN_KEYS = (
    "input_tokens",
    "request_tokens",
    "prompt_tokens",
    "inputTokens",
    "promptTokens",
)
_OUTPUT_TOKEN_KEYS = (
    "output_tokens",
    "response_tokens",
    "completion_tokens",
    "outputTokens",
    "completionTokens",
)
_TOTAL_TOKEN_KEYS = ("total_tokens", "totalTokens")


def _usage_model_settings(
    model: str | Model, model_settings: ModelSettings | None
) -> ModelSettings | None:
    """Ask OpenRouter to include native usage so token counts are not zero."""
    if not (isinstance(model, str) and model.startswith("openrouter")):
        return model_settings
    merged = dict(model_settings) if model_settings is not None else {}
    merged.setdefault("openrouter_usage", {"include": True})
    return cast("ModelSettings", merged)


def _token_count(raw: object, names: tuple[str, ...]) -> int:
    """Return the first positive alias, skipping default-zero placeholders.

    pydantic-ai ``RunUsage.input_tokens`` defaults to ``0``. Treating that as
    a real count hid OpenRouter ``request_tokens`` / ``prompt_tokens`` on the
    same object (and nested ``details``).
    """
    if isinstance(raw, dict):
        mapping = cast(dict[str, object], raw)
        values = (mapping.get(name) for name in names)
    else:
        values = (getattr(raw, name, None) for name in names)
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


def _usage_from_raw(raw: object) -> Usage:
    """Read token counts from a provider or pydantic-ai usage object."""
    if raw is None:
        return Usage(0, 0, 0)
    if isinstance(raw, Usage):
        return raw
    input_tokens = _token_count(raw, _INPUT_TOKEN_KEYS)
    output_tokens = _token_count(raw, _OUTPUT_TOKEN_KEYS)
    total_tokens = _token_count(raw, _TOTAL_TOKEN_KEYS)
    if input_tokens == 0 and output_tokens == 0:
        details = (
            cast(dict[str, object], raw).get("details")
            if isinstance(raw, dict)
            else getattr(raw, "details", None)
        )
        if details is not None:
            input_tokens = _token_count(details, _INPUT_TOKEN_KEYS)
            output_tokens = _token_count(details, _OUTPUT_TOKEN_KEYS)
            total_tokens = total_tokens or _token_count(details, _TOTAL_TOKEN_KEYS)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return Usage(input_tokens, output_tokens, total_tokens)


def _extract_usage_object(result: object) -> object:
    """Unwrap ``result.usage`` whether it is a property, method, or mapping."""
    if result is None:
        return None
    usage = getattr(result, "usage", None)
    descriptor = getattr(type(result), "usage", None)
    if usage is not None and descriptor is not None and not callable(descriptor):
        return usage
    if callable(usage):
        try:
            return usage()
        except TypeError:
            return usage
    if usage is not None:
        return usage
    return result


def _usage_from_result(result) -> Usage:
    """Build a ``Usage`` from a pydantic-ai run result or provider usage object."""
    usage = _usage_from_raw(_extract_usage_object(result))
    if usage.input_tokens or usage.output_tokens or usage.total_tokens:
        return usage
    response = getattr(result, "response", None)
    descriptor = getattr(type(result), "response", None)
    if response is None or descriptor is None or callable(descriptor):
        return usage
    nested = getattr(response, "usage", None)
    return _usage_from_raw(nested) if nested is not None else usage


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
