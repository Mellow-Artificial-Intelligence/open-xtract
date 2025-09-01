from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: int = Field(
        ..., description="The integer ID of a SPECIFIC source which justifies the answer."
    )
    quote: str = Field(
        ..., description="The VERBATIM quote from the specified source that justifies the answer."
    )


class AnnotatedAnswer(BaseModel):
    citations: list[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


def _format_docs_with_id(docs: Sequence[Document]) -> str:
    lines: list[str] = []
    for idx, d in enumerate(docs):
        snippet = d.page_content
        lines.append(f"[{idx}] {snippet}")
    return "\n\n".join(lines)


def _wrap_with_flashrank(retriever: Any) -> Any:
    """Optionally wrap a retriever with Flashrank reranker.

    Imports are done locally to keep dependency optional. Raises a helpful error
    if Flashrank integration is requested but unavailable.
    """
    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain_community.document_compressors import FlashrankRerank
    except Exception as exc:  # pragma: no cover - exercised via tests by monkeypatching
        raise RuntimeError(
            "Flashrank reranking requested, but required packages are not available."
        ) from exc

    # Instantiate FlashrankRerank in a version-agnostic way. Some versions
    # require a `client` kwarg; others accept no args. Use a dynamic call via Any
    # to avoid type mismatches across versions.
    from typing import Any, cast

    FR = cast(Any, FlashrankRerank)
    try:
        compressor = FR(client=None)
    except TypeError:
        compressor = FR()
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)


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

    # ---- Internal helpers -------------------------------------------------
    def _get_retriever(self, k: int, *, rerank: bool) -> Any:
        """Create a retriever, optionally wrapped with Flashrank reranker."""
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": k})
        if rerank:
            retriever = _wrap_with_flashrank(retriever)
        return retriever

    def _invoke_structured(self, schema: type[BaseModel], messages: list[Any]) -> BaseModel:
        """Invoke structured output for a given schema and message list."""
        return self._llm.with_structured_output(schema).invoke(messages)

    def _build_data_messages(self, question: str, docs: list[Document]) -> list[Any]:
        """Messages for initial answer generation using retrieved context."""
        docs_content = "\n\n".join(doc.page_content for doc in docs)
        return [
            SystemMessage(content="Answer the question using the provided context."),
            HumanMessage(content=f"Question: {question}\nContext:\n{docs_content}"),
        ]

    def _build_annotation_messages(
        self, question: str, data: BaseModel, docs: list[Document]
    ) -> list[Any]:
        """Messages for citation annotation of an already generated answer."""
        formatted_docs = _format_docs_with_id(docs)
        return [
            SystemMessage(
                content=f"You are given numbered sources. Use them for citations.\n\n{formatted_docs}"
            ),
            HumanMessage(content=question),
            AIMessage(content=data.model_dump_json()),
            HumanMessage(content="Annotate your answer with citations."),
        ]

    @classmethod
    def from_chroma(
        cls,
        *,
        persist_directory: str | None = None,
        collection_name: str = "open-xtract",
        embedding: Any | None = None,
    ) -> CitationRAG:
        # Import locally to keep optional deps optional at import time
        from langchain_chroma import Chroma
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        embeddings = embedding or OpenAIEmbeddings()
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        llm = ChatOpenAI()
        return cls(llm=llm, vectorstore=vs)

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        ids: Sequence[str] | None = None,
    ) -> None:
        self._vectorstore.add_texts(list(texts), metadatas=metadatas, ids=ids)

    def add_documents(self, docs: Sequence[Document], ids: Sequence[str] | None = None) -> None:
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]
        self._vectorstore.add_texts(texts, metadatas=metas, ids=ids)

    def answer(
        self,
        question: str,
        schema: type[BaseModel],
        k: int = 4,
        *,
        rerank: bool = False,
    ) -> CitationResult:
        retriever = self._get_retriever(k=k, rerank=rerank)
        docs: list[Document] = retriever.invoke(question)

        data_messages = self._build_data_messages(question, docs)
        data: BaseModel = self._invoke_structured(schema, data_messages)

        annotate_messages = self._build_annotation_messages(question, data, docs)
        annotations: AnnotatedAnswer = self._llm.with_structured_output(AnnotatedAnswer).invoke(
            annotate_messages
        )
        return CitationResult(data=data, annotations=annotations, context=docs)


__all__ = [
    "Citation",
    "AnnotatedAnswer",
    "CitationRAG",
    "CitationResult",
]
