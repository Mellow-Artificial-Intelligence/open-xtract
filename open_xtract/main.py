from __future__ import annotations

from pathlib import Path
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class OpenXtract:
    """For text extraction."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model
        self._base_url = base_url
        self._api_key = api_key
        self._llm = self._create_llm(model, base_url, api_key)

    def _create_llm(self, model: str, base_url: str | None, api_key: str | None):
        return ChatOpenAI(model=model, base_url=base_url, api_key=api_key)

    def extract(self, file_path: str | Path, schema: BaseModel):
        return self._llm.with_structured_output(schema).invoke(file_path)


__all__ = ["OpenXtract", "main"]   