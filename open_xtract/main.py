from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

# Import provider_map - try relative import first, fall back to absolute
try:
    from .provider_map import provider_map  # For when imported as a module
except ImportError:
    from provider_map import provider_map  # type: ignore[no-redef] # For when run directly
import os

from dotenv import load_dotenv

load_dotenv()


class OpenXtract:
    """For text extraction."""

    def __init__(
        self,
        model: str,
    ) -> None:
        self._model_string = model
        self._llm_parts = self._get_parts()

        self._llm = self._create_llm()

    def _get_parts(self):
        parts = self._model_string.split(":")
        self._provider = parts[0] or None
        self._model = parts[1] or None
        self._api_key = os.getenv(provider_map[self._provider]["api_key"])
        self._base_url = provider_map[self._provider]["base_url"] or None
        return self._provider, self._model, self._base_url, self._api_key

    def _create_llm(self):
        if self._provider == "anthropic":
            return ChatAnthropic(model=self._model, api_key=self._api_key)
        else:
            return ChatOpenAI(
                model=self._model, base_url=self._base_url, api_key=self._api_key
            )

    def extract(self, file_path: str | Path, schema: type[BaseModel]) -> Any:
        return self._llm.with_structured_output(schema).invoke(file_path)


__all__ = ["OpenXtract"]
