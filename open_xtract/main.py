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

# Import custom exceptions
try:
    from .exceptions import ConfigurationError, InputError, ProcessingError, ProviderError
except ImportError:
    from exceptions import (  # type: ignore[no-redef]
        ConfigurationError,
        InputError,
        ProcessingError,
        ProviderError,
    )

load_dotenv()

# Constants
MIN_API_KEY_LENGTH = 10


class OpenXtract:
    """For text extraction."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._model_string = model
        self._api_key_param = api_key
        self._base_url_param = base_url
        self._provider_param = provider
        self._llm_parts = self._get_parts()

        self._llm = self._create_llm()

    def _get_parts(self):  # noqa: PLR0912, PLR0915
        # Parse model string - support both "provider:model" and "model" formats
        if ":" in self._model_string:
            # Format: "provider:model"
            try:
                provider, model = self._model_string.split(":", 1)
            except ValueError as exc:
                msg = "Invalid model string format"
                suggestions = [
                    "Use the format 'provider:model' (e.g., 'openai:gpt-4o')",
                    "Or use 'model' with provider parameter: OpenXtract(model='gpt-4o', provider='openai')",
                    "When api_key and base_url are provided, model can be used directly without colon",
                    "Available providers: " + ", ".join(provider_map.keys()),
                    "Examples: 'openai:gpt-4o-mini', 'anthropic:claude-3-5-sonnet', 'xai:grok-beta'",
                ]
                raise ConfigurationError(msg, suggestions) from exc

            if not provider or not model:
                msg = "Model string must include both provider and model name"
                suggestions = [
                    "Provider cannot be empty",
                    "Model name cannot be empty",
                    "Use format like 'openai:gpt-4o' or 'anthropic:claude-3-5-sonnet'",
                    "Or use 'model' with provider parameter: OpenXtract(model='gpt-4o', provider='openai')",
                ]
                raise ConfigurationError(msg, suggestions)
        else:
            # Format: "model" - provider may be optional if api_key and base_url are provided
            model = self._model_string
            if not self._provider_param:
                # If api_key and base_url are provided, default to "openai" (OpenAI-compatible)
                if self._api_key_param is not None and self._base_url_param is not None:
                    provider = "openai"
                else:
                    msg = "Provider required when model string doesn't include provider"
                    suggestions = [
                        "Use the format 'provider:model' (e.g., 'openai:gpt-4o')",
                        "Or provide provider parameter: OpenXtract(model='gpt-4o', provider='openai')",
                        "Or provide api_key and base_url to use model string directly",
                        "Available providers: " + ", ".join(provider_map.keys()),
                    ]
                    raise ConfigurationError(msg, suggestions)
            else:
                provider = self._provider_param

        if not model:
            msg = "Model name cannot be empty"
            suggestions = [
                "Provide a valid model name",
                "Use format like 'openai:gpt-4o' or 'anthropic:claude-3-5-sonnet'",
                "Or use: OpenXtract(model='gpt-4o', provider='openai')",
            ]
            raise ConfigurationError(msg, suggestions)

        if provider not in provider_map:
            msg = f"Unknown provider '{provider}'"
            suggestions = [
                f"Available providers: {', '.join(provider_map.keys())}",
                "Check for typos in the provider name",
                "Ensure you're using a supported provider",
            ]
            raise ConfigurationError(msg, suggestions)

        self._provider = provider
        self._model = model

        # Priority: provided parameter > environment variable
        if self._api_key_param is not None:
            api_key = self._api_key_param
        else:
            api_key = os.getenv(provider_map[self._provider]["api_key"])

        # Validate API key
        if not api_key:
            msg = f"API key not found for provider '{provider}'"
            env_var = provider_map[self._provider]["api_key"]
            suggestions = [
                f"Set the {env_var} environment variable",
                "Or pass api_key parameter: OpenXtract(model='...', api_key='your-key')",
                "Add the API key to your .env file",
                f"Export the variable: export {env_var}=your_key_here",
            ]
            raise ProviderError(msg, provider, suggestions)

        # Basic API key validation
        if len(api_key.strip()) < MIN_API_KEY_LENGTH:
            msg = f"Invalid API key format for provider '{provider}'"
            suggestions = [
                "Ensure the API key is complete and not truncated",
                "Check for extra spaces or newline characters",
                "Verify the key matches the expected format for this provider",
            ]
            raise ProviderError(msg, provider, suggestions)

        self._api_key = api_key

        # Priority: provided parameter > provider map default
        if self._base_url_param is not None:
            base_url = self._base_url_param
        else:
            base_url = provider_map[self._provider]["base_url"] or None

        self._base_url = base_url
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
        if isinstance(data, bytes | bytearray | memoryview):
            binary = bytes(data)

            # PDF detection by header
            if binary[:5] == b"%PDF-":
                try:
                    fitz = importlib.import_module("fitz")  # PyMuPDF
                except ImportError as exc:  # pragma: no cover - import error path
                    msg = "PyMuPDF (pymupdf) is required for PDF processing"
                    suggestions = [
                        "Install with: pip install pymupdf",
                        "Or install with vision extra: pip install open-xtract[vision]",
                        "Using uv: uv add pymupdf or uv add open-xtract[vision]",
                    ]
                    raise ProcessingError(msg, suggestions) from exc

                # Render each page to PNG bytes and add to message content
                content_parts: list[str | dict[str, Any]] = [
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
                msg = "Unsupported binary input: expected image or PDF bytes"
                suggestions = [
                    "Ensure input is valid image data (PNG, JPEG, etc.)",
                    "Check that the file is not corrupted",
                    "For PDFs, ensure the file starts with '%PDF-'",
                    "Try opening the file with an image viewer first",
                ]
                raise InputError(msg, suggestions) from exc

            mime = "image/png" if format_name == "PNG" else f"image/{format_name.lower()}"
            b64 = base64.b64encode(binary).decode("utf-8")

            if self._provider == "anthropic":
                content: list[str | dict[str, Any]] = [
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
        msg = "Invalid input type for extract method"
        suggestions = [
            "Use str for text input: ox.extract('text here', Schema)",
            "Use bytes for image/PDF input: ox.extract(open('file.png', 'rb').read(), Schema)",
            "For file paths, read the file first: with open('file.pdf', 'rb') as f: ox.extract(f.read(), Schema)",
            f"Got type: {type(data).__name__}",
        ]
        raise InputError(msg, suggestions)


__all__ = ["OpenXtract"]
