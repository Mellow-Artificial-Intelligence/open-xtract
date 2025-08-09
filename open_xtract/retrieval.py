from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Type

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: int = Field(..., description="The integer ID of a SPECIFIC source which justifies the answer.")
    quote: str = Field(..., description="The VERBATIM quote from the specified source that justifies the answer.")


class AnnotatedAnswer(BaseModel):
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


def _format_docs_with_id(docs: Sequence[Document]) -> str:
    lines: list[str] = []
    for idx, d in enumerate(docs):
        snippet = d.page_content
        lines.append(f"[{idx}] {snippet}")
    return "\n\n".join(lines)


@dataclass
class CitationResult:
    data: BaseModel
    annotations: AnnotatedAnswer
    context: list[Document]


class CitationRAG:
    """RAG with citation post-processing using structured output.

    This follows the pattern in LangChain's QA citation guide and returns an
    answer plus structured citations extracted from the retrieved context.
    """

    def __init__(self, llm: Any, vectorstore: Any):
        self._llm = llm
        self._vectorstore = vectorstore

    @classmethod
    def from_chroma(
        cls,
        *,
        persist_directory: str | None = None,
        collection_name: str = "open-xtract",
        embedding: Any | None = None,
    ) -> "CitationRAG":
        # Import locally to keep optional deps optional at import time
        from langchain_chroma import Chroma  # type: ignore
        from langchain_openai import OpenAIEmbeddings  # type: ignore
        from langchain_openai import ChatOpenAI  # type: ignore

        embeddings = embedding or OpenAIEmbeddings()
        vs = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory)
        llm = ChatOpenAI()
        return cls(llm=llm, vectorstore=vs)

    def add_texts(self, texts: Sequence[str], metadatas: Sequence[dict] | None = None, ids: Sequence[str] | None = None) -> None:
        self._vectorstore.add_texts(list(texts), metadatas=metadatas, ids=ids)

    def add_documents(self, docs: Sequence[Document], ids: Sequence[str] | None = None) -> None:
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]
        self._vectorstore.add_texts(texts, metadatas=metas, ids=ids)

    def answer(
        self,
        question: str,
        schema: Type[BaseModel],
        k: int = 4,
        *,
        rerank: bool = False,
    ) -> CitationResult:
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": k})
        if rerank:
            retriever = _wrap_with_flashrank(retriever)
        docs: list[Document] = retriever.invoke(question)

        docs_content = "\n\n".join(doc.page_content for doc in docs)
        messages: list[Any] = [
            SystemMessage(content="Answer the question using the provided context."),
            HumanMessage(content=f"Question: {question}\nContext:\n{docs_content}"),
        ]
        structured_data_llm = self._llm.with_structured_output(schema)
        data: BaseModel = structured_data_llm.invoke(messages)  # type: ignore

        formatted_docs = _format_docs_with_id(docs)
        structured_llm = self._llm.with_structured_output(AnnotatedAnswer)
        annotate_messages: list[Any] = [
            SystemMessage(content=f"You are given numbered sources. Use them for citations.\n\n{formatted_docs}"),
            HumanMessage(content=question),
            AIMessage(content=data.model_dump_json()),
            HumanMessage(content="Annotate your answer with citations."),
        ]
        annotations: AnnotatedAnswer = structured_llm.invoke(annotate_messages)
        return CitationResult(data=data, annotations=annotations, context=docs)


__all__ = [
    "Citation",
    "AnnotatedAnswer",
    "CitationRAG",
    "CitationResult",
]


def _wrap_with_flashrank(retriever: Any) -> Any:
    """Optionally wrap a retriever with Flashrank reranker.

    Imports are done locally to keep dependency optional. Raises a helpful error
    if Flashrank integration is requested but unavailable.
    """
    try:
        from langchain.retrievers import ContextualCompressionRetriever  # type: ignore
        from langchain_community.document_compressors import FlashrankRerank  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised via tests by monkeypatching
        raise RuntimeError(
            "Flashrank reranking requested, but required packages are not available."
        ) from exc

    compressor = FlashrankRerank()
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)


