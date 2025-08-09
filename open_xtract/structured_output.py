"""Utilities for generating structured outputs with LangChain chat models.

This module provides a light abstraction over LangChain's
`with_structured_output` so callers can generate structured outputs from:

- Pydantic BaseModel types
- JSON Schemas (as dict or JSON string)
- Freeform prompts or LCEL prompts

The goal is to offer a small, focused API with sensible defaults while keeping
call sites concise and readable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union, overload

try:
    # LangChain >= 0.3
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable
except Exception:  # pragma: no cover - keep imports flexible across envs
    BaseChatModel = object  # type: ignore[misc,assignment]
    ChatPromptTemplate = object  # type: ignore[misc,assignment]
    Runnable = object  # type: ignore[misc,assignment]

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[misc,assignment]


SchemaLike = Union[Type[BaseModel], Dict[str, Any], str]
PromptLike = Union[str, "ChatPromptTemplate", "Runnable", Sequence[Tuple[str, str]]]


class StructuredOutputGenerator:
    """Generate structured outputs from a chat model using a given schema.

    This wraps `llm.with_structured_output(...)` and provides a small helper to
    prepare prompts/messages in a consistent way.

    Parameters
    - llm: LangChain chat model instance
    - schema: Pydantic model class, JSON schema dict, or JSON schema string
    - name: Optional human-friendly name for the schema/tool (passed through)
    - method: Optional. When provided, should be one of {"json_mode", "function_calling"}.
              If "auto" is passed, it will be ignored for compatibility with providers
              that don't accept it.
    - strict: Whether validation should be strict (passed through)
    - kwargs: Forwarded to `with_structured_output`
    """

    def __init__(
        self,
        llm: Any,
        schema: SchemaLike,
        *,
        name: Optional[str] = None,
        method: Optional[str] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        self._llm: Any = llm
        self._schema: SchemaLike = self._normalize_schema(schema)
        call_kwargs: Dict[str, Any] = {"name": name, "strict": strict, **kwargs}
        # Some providers (e.g., langchain-openai>=0.3.x) only accept "json_mode" or
        # "function_calling" and will raise on "auto". Skip passing method when it's
        # None or explicitly "auto" to let the provider choose sensible defaults.
        if method and method != "auto":
            call_kwargs["method"] = method
        self._structured = self._llm.with_structured_output(self._schema, **call_kwargs)

    # ---- Public API -----------------------------------------------------
    def invoke(
        self,
        prompt: PromptLike,
        variables: Optional[Mapping[str, Any]] = None,
        *,
        config: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Invoke the structured model with the provided prompt.

        - prompt may be:
          - a plain user string
          - a `ChatPromptTemplate`
          - a `Runnable` producing messages
          - a sequence of (role, content) message tuples
        - variables are used to format LCEL prompts/templates
        - config is forwarded to the runnable's `.invoke(..., config=...)`
        """
        messages = self._prepare_messages(prompt, variables)
        if config is not None:
            return self._structured.invoke(messages, config=config)
        return self._structured.invoke(messages)

    async def ainvoke(
        self,
        prompt: PromptLike,
        variables: Optional[Mapping[str, Any]] = None,
        *,
        config: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        messages = self._prepare_messages(prompt, variables)
        if config is not None:
            return await self._structured.ainvoke(messages, config=config)
        return await self._structured.ainvoke(messages)

    # ---- Helpers --------------------------------------------------------
    @staticmethod
    def _normalize_schema(schema: SchemaLike) -> SchemaLike:
        """Allow JSON schema as dict or JSON string; pass Pydantic through."""
        if isinstance(schema, str):
            # Attempt to parse JSON strings to dicts for clearer errors downstream
            try:
                return json.loads(schema)
            except json.JSONDecodeError:
                # If it's not valid JSON, leave as-is and let LangChain raise
                return schema
        return schema

    @staticmethod
    def _prepare_messages(
        prompt: PromptLike, variables: Optional[Mapping[str, Any]]
    ) -> Any:
        # ChatPromptTemplate
        if isinstance(prompt, ChatPromptTemplate):  # type: ignore[arg-type]
            return prompt.invoke(variables or {})

        # Generic Runnable (e.g., composed LCEL graph)
        if hasattr(prompt, "invoke") and not isinstance(
            prompt, (str, ChatPromptTemplate)
        ):
            # Treat as Runnable chain that yields messages
            return prompt.invoke(variables or {})  # type: ignore[no-any-return]

        # Sequence of (role, content) tuples
        if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
            return list(prompt)

        # Plain text prompt
        if isinstance(prompt, str):
            if variables:
                try:
                    formatted = prompt.format(**variables)
                except Exception:
                    # If formatting fails, fall back to raw prompt
                    formatted = prompt
            else:
                formatted = prompt
            return [("human", formatted)]

        # Last resort: pass through
        return prompt


__all__ = ["StructuredOutputGenerator"]


