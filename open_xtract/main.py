from __future__ import annotations

import base64
import io
from typing import Iterable, Type

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from .retrieval import CitationRAG, CitationResult


class OpenXtract:
    """Concise facade for extracting structured data from PDFs, images, and text."""

    def __init__(
        self,
        model: str = "gpt-5-nano",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        *,
        max_tokens_image_description: int = 1024,
        **llm_kwargs,
    ) -> None:
        self._model_name = model
        self._base_url = base_url
        self._api_key = api_key
        self._llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, **llm_kwargs)
        self._max_tokens_image_description = max_tokens_image_description
        self._citation_rag: CitationRAG | None = None

    # PDF
    def extract_pdf(
        self,
        pdf_path: str,
        schema: Type[BaseModel],
        instruction: str = "Extract the relevant information from the PDF",
        *,
        stream: bool = False,
    ) -> BaseModel | Iterable[BaseModel]:
        loader = PyPDFLoader(
            pdf_path,
            mode="page",
            images_inner_format="markdown-img",
            images_parser=LLMImageBlobParser(
                model=ChatOpenAI(
                    model=self._model_name,
                    base_url=self._base_url,
                    api_key=self._api_key,
                    max_tokens=self._max_tokens_image_description,
                )
            ),
        )
        docs = loader.load()
        structured_llm = self._llm.with_structured_output(schema)
        prompt = (
            "Please follow the instructions below to extract the relevant information from the PDF.\n"
            f"{instruction}\n\nThe PDF is as follows:\n{docs}\n"
        )
        if stream:
            return structured_llm.stream([HumanMessage(content=prompt)])
        return structured_llm.invoke([HumanMessage(content=prompt)])

    # Image
    def extract_image(
        self,
        image_path_or_url: str,
        schema: Type[BaseModel],
        instruction: str = "Extract the relevant information from the image",
        *,
        stream: bool = False,
    ) -> BaseModel | Iterable[BaseModel]:
        message_content = [
            {"type": "text", "text": instruction},
        ]
        if image_path_or_url.startswith(("http://", "https://")):
            message_content.append({"type": "image_url", "image_url": {"url": image_path_or_url}})
        else:
            image_bytes = io.BytesIO()
            with open(image_path_or_url, "rb") as img_file:
                image_bytes.write(img_file.read())
            img_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})

        structured_llm = self._llm.with_structured_output(schema)
        if stream:
            return structured_llm.stream([HumanMessage(content=message_content)])
        return structured_llm.invoke([HumanMessage(content=message_content)])

    # Text
    def extract_text(
        self,
        text: str,
        schema: Type[BaseModel],
        *,
        stream: bool = False,
    ) -> BaseModel | Iterable[BaseModel]:
        structured_llm = self._llm.with_structured_output(schema)
        if stream:
            return structured_llm.stream([HumanMessage(content=text)])
        return structured_llm.invoke([HumanMessage(content=text)])

    # Retrieval with optional reranking and citations
    def retrieve(
        self,
        question: str,
        *,
        texts: list[str] | None = None,
        docs: list | None = None,
        schema: Type[BaseModel],
        k: int = 4,
        rerank: bool = False,
    ) -> CitationResult:
        """Retrieve, answer, and annotate with citations.

        Provide either raw `texts` or pre-built `docs` (LangChain Documents). Data will
        be embedded into an in-memory vector store for this call.
        """
        from langchain_chroma import Chroma  # type: ignore
        from langchain_openai import OpenAIEmbeddings  # type: ignore
        from langchain_core.documents import Document  # type: ignore

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(collection_name="open-xtract-semantic", embedding_function=embeddings)
        rag = CitationRAG(llm=self._llm, vectorstore=vectorstore)
        if texts:
            rag.add_texts(texts)
        if docs:
            # type: ignore[arg-type]
            rag.add_documents(docs)  # expects Sequence[Document]
        return rag.answer(question, schema=schema, k=k, rerank=rerank)


def main() -> None:
    # Minimal CLI placeholder to keep the entrypoint intact.
    print("open-xtract: CLI not configured. Import and use `OpenXtract` programmatically.")


__all__ = ["OpenXtract", "main"]
