from __future__ import annotations

from pathlib import Path
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
try:
    from .provider_map import provider_map  # For when imported as a module
except ImportError:
    from provider_map import provider_map  # For when run directly
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
        self._provider = parts[0]
        self._model = parts[1]
        self._api_key = os.getenv(provider_map[self._provider]["api_key"])
        self._base_url = provider_map[self._provider]["base_url"]
        return self._provider, self._model, self._base_url, self._api_key

    def _create_llm(self):
        return ChatOpenAI(model=self._llm_parts[1], base_url=self._llm_parts[2], api_key=self._llm_parts[3])

    def extract(self, file_path: str | Path, schema: BaseModel):
        return self._llm.with_structured_output(schema).invoke(file_path)


__all__ = ["OpenXtract", "main"]