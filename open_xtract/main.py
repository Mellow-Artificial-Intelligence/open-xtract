from __future__ import annotations

import base64
import io
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, SecretStr
from typing import Any, cast
from langchain_core.documents import Document

from .retrieval import CitationRAG, CitationResult


class OpenXtract:
    """Concise facade for extracting structured data from PDFs, images, and text."""

    def __init__(
        self,
        model: str = "gpt-5-nano",
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        max_tokens_image_description: int = 1024,
        **llm_kwargs,
    ) -> None:
        self._model_name = model
        self._base_url = base_url
        self._api_key = api_key
        self._llm = self._create_llm(model, base_url, api_key, **llm_kwargs)
        self._max_tokens_image_description = max_tokens_image_description
        self._citation_rag: CitationRAG | None = None

        # File type mappings
        self._pdf_extensions = {".pdf"}
        self._image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

    def _create_llm(self, model: str, base_url: str | None, api_key: str | None, **llm_kwargs):
        """Create the appropriate LLM instance based on the model name."""
        model_lower = model.lower()
        
        def _as_secret(value: str | None) -> SecretStr | None:
            return SecretStr(value) if value is not None else None

        if "claude" in model_lower or "anthropic" in model_lower:
            try:
                from langchain_anthropic import ChatAnthropic

                anth_kwargs: dict[str, Any] = {"model": model, **llm_kwargs}
                if api_key is not None:
                    anth_kwargs["api_key"] = _as_secret(api_key)
                return ChatAnthropic(**anth_kwargs)
            except ImportError as err:
                raise ImportError(
                    "langchain-anthropic is required for Claude models. Install with: pip install langchain-anthropic"
                ) from err

        elif "gemini" in model_lower or "google" in model_lower:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **llm_kwargs)
            except ImportError as err:
                raise ImportError(
                    "langchain-google-genai is required for Google models. Install with: pip install langchain-google-genai"
                ) from err

        else:
            # Default to OpenAI (includes OpenAI-compatible endpoints)
            try:
                from langchain_openai import ChatOpenAI

                base_url = base_url or "https://api.openai.com/v1"
                return ChatOpenAI(model=model, base_url=base_url, api_key=_as_secret(api_key), **llm_kwargs)
            except ImportError as err:
                raise ImportError(
                    "langchain-openai is required for OpenAI models. Install with: pip install langchain-openai"
                ) from err

    def extract(
        self,
        input_data: str,
        schema: type[BaseModel],
        instruction: str = "Extract the relevant information",
        *,
        stream: bool = False,
    ) -> BaseModel | Iterable[BaseModel]:
        """Auto-route extraction based on input type (file extension, URL, or raw text)."""
        input_type = self._detect_input_type(input_data)

        if input_type == "pdf":
            return self._extract_pdf(input_data, schema, instruction, stream=stream)
        elif input_type in ["image", "image_url", "pdf_url"]:
            return self._extract_image(input_data, schema, instruction, stream=stream)
        else:  # text
            return self._extract_text(input_data, schema, stream=stream)

    # Retrieval with optional reranking and citations
    def retrieve(
        self,
        question: str,
        *,
        texts: list[str] | None = None,
        docs: list[Document] | None = None,
        schema: type[BaseModel],
        k: int = 4,
        rerank: bool = False,
    ) -> CitationResult:
        """Retrieve, answer, and annotate with citations.

        Provide either raw `texts` or pre-built `docs` (LangChain Documents). Data will
        be embedded into an in-memory vector store for this call.
        """
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(collection_name="open-xtract-semantic", embedding_function=embeddings)
        rag = CitationRAG(llm=self._llm, vectorstore=vectorstore)
        if texts:
            rag.add_texts(texts)
        if docs:
            rag.add_documents(docs)
        return rag.answer(question, schema=schema, k=k, rerank=rerank)

    def _detect_input_type(self, input_data: str) -> str:
        """Detect the type of input data (pdf, image, url, or text)."""
        # Check if it's a URL
        if input_data.startswith(("http://", "https://")):
            parsed = urlparse(input_data)
            if parsed.path:
                ext = Path(parsed.path).suffix.lower()
                if ext in self._pdf_extensions:
                    return "pdf_url"
                elif ext in self._image_extensions:
                    return "image_url"
            return "image_url"  # Default URLs to image handling

        # Check for file extensions
        ext = Path(input_data).suffix.lower()
        if ext in self._pdf_extensions or ext in self._image_extensions:
            # If it has a recognized extension, it should be a valid file
            if not os.path.isfile(input_data):
                raise FileNotFoundError(f"File not found: {input_data}")
            return "pdf" if ext in self._pdf_extensions else "image"

        # If no extension and not a URL, treat as raw text
        return "text"

    # ---- Internal helpers -------------------------------------------------
    def _run_structured(
        self,
        schema: type[BaseModel],
        message: HumanMessage,
        *,
        stream: bool,
    ) -> BaseModel | Iterable[BaseModel]:
        """Invoke or stream a structured call uniformly."""
        structured_llm: Any = self._llm.with_structured_output(schema)
        if stream:
            return cast(Iterable[BaseModel], structured_llm.stream([message]))
        return cast(BaseModel, structured_llm.invoke([message]))

    def _build_image_message_content(self, image_path_or_url: str, instruction: str) -> list[dict[str, Any] | str]:
        """Build multi-modal message content for an image path or URL."""
        message_content: list[dict[str, Any] | str] = [
            {"type": "text", "text": instruction},
        ]
        if image_path_or_url.startswith(("http://", "https://")):
            message_content.append({"type": "image_url", "image_url": {"url": image_path_or_url}})
        else:
            # Keep jpeg data URL prefix for compatibility with downstream providers/tests
            with open(image_path_or_url, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            message_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            )
        return message_content

    def _build_pdf_prompt(self, docs: Sequence[Any], instruction: str) -> str:
        """Create the prompt string for PDF-based extraction."""
        return (
            "Please follow the instructions below to extract the relevant information from the PDF.\n"
            f"{instruction}\n\nThe PDF is as follows:\n{docs}\n"
        )

    def _extract_pdf(
        self,
        pdf_path: str,
        schema: type[BaseModel],
        instruction: str,
        *,
        stream: bool = False,
    ) -> BaseModel | Iterable[BaseModel]:
        # Create a separate LLM instance for image parsing with max_tokens
        image_llm = self._create_llm(
            self._model_name,
            self._base_url,
            self._api_key,
            max_tokens=self._max_tokens_image_description,
        )
        loader = PyPDFLoader(
            pdf_path,
            mode="page",
            images_inner_format="markdown-img",
            images_parser=LLMImageBlobParser(model=image_llm),
        )
        docs = loader.load()
        prompt = self._build_pdf_prompt(docs, instruction)
        return self._run_structured(schema, HumanMessage(content=prompt), stream=stream)

    def _extract_image(
        self,
        image_path_or_url: str,
        schema: type[BaseModel],
        instruction: str,
        *,
        stream: bool = False,
    ) -> BaseModel | Iterable[BaseModel]:
        message_content = self._build_image_message_content(image_path_or_url, instruction)
        # HumanMessage expects content as str or list of rich content items
        content_list: list[str | dict[str, Any]] = [item for item in message_content]
        return self._run_structured(schema, HumanMessage(content=content_list), stream=stream)

    def _extract_text(
        self,
        text: str,
        schema: type[BaseModel],
        *,
        stream: bool = False,
    ) -> BaseModel | Iterable[BaseModel]:
        return self._run_structured(schema, HumanMessage(content=text), stream=stream)

    


def main() -> None:
    # Minimal CLI placeholder to keep the entrypoint intact.
    print("open-xtract: CLI not configured. Import and use `OpenXtract` programmatically.")


__all__ = ["OpenXtract", "main"]
