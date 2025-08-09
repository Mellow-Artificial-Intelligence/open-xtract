from __future__ import annotations

from typing import Any, Iterable

import open_xtract.main as ox_module
from open_xtract import OpenXtract
from pydantic import BaseModel
from open_xtract.retrieval import AnnotatedAnswer


class _FakeStructured:
    def __init__(self, schema: type[BaseModel]) -> None:
        self.schema = schema

    def invoke(self, messages: list[Any]) -> BaseModel:
        _FakeChatOpenAI.last_invocations.append(("invoke", self.schema, messages))
        if self.schema is AnnotatedAnswer:
            return AnnotatedAnswer(citations=[])
        return self.schema.model_validate(
            {k: 0 if v.annotation in (int, float) else "ok" for k, v in self.schema.model_fields.items()}
        )

    def stream(self, messages: list[Any]) -> Iterable[BaseModel]:
        _FakeChatOpenAI.last_invocations.append(("stream", self.schema, messages))
        if self.schema is AnnotatedAnswer:
            yield AnnotatedAnswer(citations=[])
        else:
            yield self.schema.model_validate(
                {k: 0 if v.annotation in (int, float) else "ok" for k, v in self.schema.model_fields.items()}
            )


class _FakeChatOpenAI:
    last_invocations: list[tuple[str, type[BaseModel], list[Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        pass

    def with_structured_output(self, schema: type[BaseModel]) -> _FakeStructured:
        return _FakeStructured(schema)


class _FakePDFLoader:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def load(self) -> list[str]:
        return ["DOC1", "DOC2"]


class _FakeLLMImageBlobParser:  # noqa: D401
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _Schema(BaseModel):
    text: str


def test_extract_text_invoke(monkeypatch: Any) -> None:
    monkeypatch.setattr(ox_module, "ChatOpenAI", _FakeChatOpenAI)
    ox = OpenXtract()
    res = ox.extract_text("hello", _Schema)
    assert isinstance(res, _Schema)
    assert res.text == "ok"
    kind, schema, messages = _FakeChatOpenAI.last_invocations[-1]
    assert kind == "invoke"
    assert schema is _Schema
    assert isinstance(messages, list) and len(messages) == 1


def test_extract_text_stream(monkeypatch: Any) -> None:
    monkeypatch.setattr(ox_module, "ChatOpenAI", _FakeChatOpenAI)
    ox = OpenXtract()
    chunks = list(ox.extract_text("hello", _Schema, stream=True))
    assert len(chunks) == 1 and isinstance(chunks[0], _Schema)
    kind, _, _ = _FakeChatOpenAI.last_invocations[-1]
    assert kind == "stream"


def test_extract_image_stream(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setattr(ox_module, "ChatOpenAI", _FakeChatOpenAI)
    img_path = tmp_path / "img.bin"
    img_path.write_bytes(b"bytes")
    ox = OpenXtract()
    chunks = list(ox.extract_image(str(img_path), _Schema, stream=True))
    assert len(chunks) == 1 and isinstance(chunks[0], _Schema)


def test_extract_pdf_stream(monkeypatch: Any) -> None:
    monkeypatch.setattr(ox_module, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(ox_module, "PyPDFLoader", _FakePDFLoader)
    monkeypatch.setattr(ox_module, "LLMImageBlobParser", _FakeLLMImageBlobParser)
    ox = OpenXtract()
    chunks = list(ox.extract_pdf("/path/to/doc.pdf", _Schema, stream=True))
    assert len(chunks) == 1 and isinstance(chunks[0], _Schema)


def test_retrieve_with_texts(monkeypatch: Any) -> None:
    # Fakes for embeddings and vector store
    class _FakeEmbeddings: ...

    class _FakeRetriever:
        def __init__(self, docs: list[Any]) -> None:
            self._docs = docs

        def invoke(self, _q: str):
            return self._docs

    class _FakeChroma:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._texts: list[str] = []

        def add_texts(self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None) -> None:
            self._texts.extend(texts)

        def as_retriever(self, search_kwargs: dict | None = None) -> _FakeRetriever:
            docs = [type("D", (), {"page_content": t, "metadata": {}}) for t in self._texts]
            return _FakeRetriever(docs)

    monkeypatch.setattr(ox_module, "ChatOpenAI", _FakeChatOpenAI)
    # Inject fake modules so inner imports resolve
    import types, sys
    fake_openai_mod = types.SimpleNamespace(OpenAIEmbeddings=_FakeEmbeddings)
    fake_chroma_mod = types.SimpleNamespace(Chroma=_FakeChroma)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_openai_mod)
    monkeypatch.setitem(sys.modules, "langchain_chroma", fake_chroma_mod)

    class Out(BaseModel):
        val: int

    ox = OpenXtract()
    result = ox.retrieve(
        question="q",
        texts=["a", "b"],
        schema=Out,
        k=1,
        rerank=False,
    )
    assert isinstance(result.data, Out)


def test_retrieve_with_docs(monkeypatch: Any) -> None:
    class _FakeEmbeddings: ...

    class _FakeRetriever:
        def __init__(self, docs: list[Any]) -> None:
            self._docs = docs

        def invoke(self, _q: str):
            return self._docs

    class _FakeChroma:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._texts: list[str] = []

        def add_texts(self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None) -> None:
            self._texts.extend(texts)

        def as_retriever(self, search_kwargs: dict | None = None) -> _FakeRetriever:
            docs = [type("D", (), {"page_content": t, "metadata": {}}) for t in self._texts]
            return _FakeRetriever(docs)

    # Simple Document shim for input
    class _Doc:
        def __init__(self, page_content: str, metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    monkeypatch.setattr(ox_module, "ChatOpenAI", _FakeChatOpenAI)
    import types, sys
    fake_openai_mod = types.SimpleNamespace(OpenAIEmbeddings=_FakeEmbeddings)
    fake_chroma_mod = types.SimpleNamespace(Chroma=_FakeChroma)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_openai_mod)
    monkeypatch.setitem(sys.modules, "langchain_chroma", fake_chroma_mod)

    class Out(BaseModel):
        text: str

    ox = OpenXtract()
    result = ox.retrieve(
        question="q",
        docs=[_Doc("x"), _Doc("y")],
        schema=Out,
        k=2,
        rerank=False,
    )
    assert isinstance(result.data, Out)


def test_extract_image_with_url(monkeypatch: Any) -> None:
    monkeypatch.setattr(ox_module, "ChatOpenAI", _FakeChatOpenAI)
    ox = OpenXtract()
    res = ox.extract_image("https://example.com/img.png", _Schema)
    assert isinstance(res, _Schema)


def test_extract_image_with_file(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setattr(ox_module, "ChatOpenAI", _FakeChatOpenAI)
    img_path = tmp_path / "img.bin"
    img_path.write_bytes(b"not-a-real-image-but-ok")
    ox = OpenXtract()
    res = ox.extract_image(str(img_path), _Schema)
    assert isinstance(res, _Schema)


def test_extract_pdf_invoke(monkeypatch: Any) -> None:
    monkeypatch.setattr(ox_module, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(ox_module, "PyPDFLoader", _FakePDFLoader)
    monkeypatch.setattr(ox_module, "LLMImageBlobParser", _FakeLLMImageBlobParser)
    ox = OpenXtract()
    res = ox.extract_pdf("/path/to/doc.pdf", _Schema)
    assert isinstance(res, _Schema)
    # Ensure prompt contains DOC1/DOC2 as part of content
    _, _, messages = _FakeChatOpenAI.last_invocations[-1]
    assert any("DOC1" in (msg.content if hasattr(msg, "content") else str(msg)) for msg in messages)


