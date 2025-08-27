from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from open_xtract import OpenXtract
from pydantic import BaseModel


class _FakeStructured:
    def __init__(self, schema: type[BaseModel]) -> None:
        self.schema = schema

    def invoke(self, _messages: list[Any]) -> BaseModel:
        if self.schema.__name__ == "AnnotatedAnswer":
            # Minimal structure for citations
            from open_xtract.retrieval import AnnotatedAnswer

            return AnnotatedAnswer(citations=[])
        return self.schema.model_validate(
            {
                k: 0 if v.annotation in (int, float) else "ok"
                for k, v in self.schema.model_fields.items()
            }
        )


class _FakeLLM:
    def with_structured_output(self, schema: type[BaseModel]) -> _FakeStructured:
        return _FakeStructured(schema)


class _FakeRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def invoke(self, _q: str):
        return self._docs


class _FakeVectorStore:
    def __init__(self) -> None:
        self._texts: list[str] = []

    def add_texts(
        self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None
    ) -> None:
        self._texts.extend(texts)

    def as_retriever(self, search_kwargs: dict | None = None) -> _FakeRetriever:
        docs = [Document(page_content=t) for t in self._texts]
        return _FakeRetriever(docs)


def test_openxtract_retrieve_with_rerank_true(monkeypatch: Any) -> None:
    # Replace the _create_llm with a fake and inject fake vectorstore classes
    monkeypatch.setattr(OpenXtract, "_create_llm", lambda *args, **kwargs: _FakeLLM())

    # Patch the imports inside OpenXtract.retrieve to use our fakes
    import sys
    import types

    fake_openai_mod = types.SimpleNamespace(OpenAIEmbeddings=object)
    fake_chroma_mod = types.SimpleNamespace(Chroma=lambda **kwargs: _FakeVectorStore())
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_openai_mod)
    monkeypatch.setitem(sys.modules, "langchain_chroma", fake_chroma_mod)

    class Out(BaseModel):
        text: str

    ox = OpenXtract()
    # We will pass texts and set rerank=True which will trigger _wrap_with_flashrank in CitationRAG
    # To avoid external dependency, monkeypatch the internal function to a no-op wrapper
    from open_xtract import main as ox_main
    from open_xtract import retrieval as ox_retrieval

    monkeypatch.setattr(ox_retrieval, "_wrap_with_flashrank", lambda r: r)

    res = ox.retrieve(question="q", texts=["hello", "world"], schema=Out, k=1, rerank=True)
    assert isinstance(res.data, Out)
    assert len(res.context) >= 1

