from __future__ import annotations

import base64
import importlib
import io
import os
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel

# Import provider_map - try relative import first, fall back to absolute
try:
    from .provider_map import provider_map  # For when imported as a module
except ImportError:
    from provider_map import provider_map  # type: ignore[no-redef] # For when run directly

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
        try:
            provider, model = self._model_string.split(":", 1)
        except ValueError as exc:  # Not enough parts to unpack
            msg = "model string must be in the format 'provider:model'"
            raise ValueError(msg) from exc

        if not provider or not model:
            raise ValueError("model string must include both provider and model")

        if provider not in provider_map:
            raise ValueError(f"unknown provider '{provider}'")

        self._provider = provider
        self._model = model
        self._api_key = os.getenv(provider_map[self._provider]["api_key"])
        self._base_url = provider_map[self._provider]["base_url"] or None
        return self._provider, self._model, self._base_url, self._api_key

    def _create_llm(self):
        if self._provider == "anthropic":
            return ChatAnthropic(model=self._model, api_key=self._api_key)
        else:
            return ChatOpenAI(model=self._model, base_url=self._base_url, api_key=self._api_key)

    def extract(self, data: str | bytes, schema: type[BaseModel]) -> Any:
        """Extract structured data from text, images, or PDFs.

        Requirements:
        - Images must be provided as raw bytes (user reads file first)
        - PDFs must be provided as raw bytes; each page is rendered to an image and sent
        - Plain text is accepted as `str`
        """

        # Text input path: pass through unchanged
        if isinstance(data, str):
            return self._llm.with_structured_output(schema).invoke(data)

        # Support bytes-like inputs
        if isinstance(data, (bytes, bytearray, memoryview)):
            binary = bytes(data)

            # PDF detection by header
            if binary[:5] == b"%PDF-":
                try:
                    fitz = importlib.import_module("fitz")  # PyMuPDF
                except Exception as exc:  # pragma: no cover - import error path
                    msg = (
                        "PyMuPDF (pymupdf) is required for PDF page rendering. "
                        "Install with `pip install pymupdf` or include the vision extra."
                    )
                    raise RuntimeError(msg) from exc

                # Render each page to PNG bytes and add to message content
                content_parts: list[dict[str, Any]] = [
                    {"type": "text", "text": "Extract the structured data per the provided schema."}
                ]
                with fitz.open(stream=binary, filetype="pdf") as doc:
                    for page in doc:
                        pix = page.get_pixmap(dpi=200)
                        png_bytes = pix.tobytes("png")
                        b64 = base64.b64encode(png_bytes).decode("utf-8")
                        if self._provider == "anthropic":
                            content_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b64,
                                    },
                                }
                            )
                        else:
                            data_url = f"data:image/png;base64,{b64}"
                            content_parts.append(
                                {"type": "image_url", "image_url": {"url": data_url}}
                            )

                return self._llm.with_structured_output(schema).invoke(
                    [HumanMessage(content=content_parts)]
                )

            # Image bytes path: verify and send as multimodal
            try:
                with Image.open(io.BytesIO(binary)) as img:
                    format_name = (img.format or "PNG").upper()
            except Exception as exc:
                raise ValueError("Unsupported binary input: expected image or PDF bytes") from exc

            mime = "image/png" if format_name == "PNG" else f"image/{format_name.lower()}"
            b64 = base64.b64encode(binary).decode("utf-8")

            if self._provider == "anthropic":
                content = [
                    {
                        "type": "text",
                        "text": "Extract the structured data per the provided schema.",
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": b64},
                    },
                ]
            else:
                data_url = f"data:{mime};base64,{b64}"
                content = [
                    {
                        "type": "text",
                        "text": "Extract the structured data per the provided schema.",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]

            return self._llm.with_structured_output(schema).invoke([HumanMessage(content=content)])

        # Reject file paths or other unsupported types to keep API strict
        raise TypeError("extract expects `str` for text or `bytes` for image/PDF content")


__all__ = ["OpenXtract"]
