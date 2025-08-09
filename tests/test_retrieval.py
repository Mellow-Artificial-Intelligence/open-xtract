from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from open_xtract.retrieval import AnnotatedAnswer, CitationRAG
from pydantic import BaseModel


class _FakeRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def invoke(self, _q: str):
        return self._docs


class _FakeVectorStore:
    def __init__(self) -> None:
        self._texts: list[str] = []

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None) -> None:
        self._texts.extend(texts)

    def as_retriever(self, search_kwargs: dict | None = None) -> _FakeRetriever:
        docs = [Document(page_content=t) for t in self._texts]
        return _FakeRetriever(docs)


class _FakeStructured:
    def __init__(self, schema: type[BaseModel]) -> None:
        self._schema = schema

    def invoke(self, _messages: list[Any]) -> BaseModel:
        if self._schema is AnnotatedAnswer:
            return AnnotatedAnswer(citations=[])
        return self._schema.model_validate(
            {k: 0 if v.annotation in (int, float) else "ok" for k, v in self._schema.model_fields.items()}
        )


class _FakeLLM:
    def with_structured_output(self, schema: type[BaseModel]) -> _FakeStructured:
        return _FakeStructured(schema)


def test_citation_rag_answer() -> None:
    vs = _FakeVectorStore()
    vs.add_texts(["The cheetah is capable of running at 93 to 104 km/h."])
    rag = CitationRAG(llm=_FakeLLM(), vectorstore=vs)

    class SpeedAnswer(BaseModel):
        speed_kmh: float
        text: str

    result = rag.answer("How fast are cheetahs?", schema=SpeedAnswer, k=1)
    assert isinstance(result.data, SpeedAnswer)
    assert isinstance(result.annotations, AnnotatedAnswer)
    assert len(result.context) == 1


def test_citation_rag_rerank_branch(monkeypatch: Any) -> None:
    # Force rerank path without importing real flashrank
    def fake_wrap(retriever: Any) -> Any:
        return retriever

    monkeypatch.setattr("open_xtract.retrieval._wrap_with_flashrank", fake_wrap)
    vs = _FakeVectorStore()
    vs.add_texts(["A", "B"])
    rag = CitationRAG(llm=_FakeLLM(), vectorstore=vs)

    class Out(BaseModel):
        n: int
        text: str

    res = rag.answer("q", schema=Out, k=1, rerank=True)
    assert isinstance(res.data, Out)


