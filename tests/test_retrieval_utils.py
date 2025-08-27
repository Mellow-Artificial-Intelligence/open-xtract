from __future__ import annotations

import types
import sys

import pytest
from langchain_core.documents import Document

from open_xtract.retrieval import _format_docs_with_id, _wrap_with_flashrank


def test_format_docs_with_id_basic() -> None:
    docs = [Document(page_content="A"), Document(page_content="B")]
    formatted = _format_docs_with_id(docs)
    # Expect each line to be prefixed with an index in brackets
    assert "[0] A" in formatted
    assert "[1] B" in formatted


def test_wrap_with_flashrank_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure that attempting to wrap without required deps raises a clear RuntimeError
    # Simulate missing symbol by injecting an empty module for langchain.retrievers
    fake_retrievers = types.ModuleType("langchain.retrievers")
    monkeypatch.setitem(sys.modules, "langchain.retrievers", fake_retrievers)

    with pytest.raises(RuntimeError) as excinfo:
        _wrap_with_flashrank(retriever=object())

    assert "Flashrank reranking requested" in str(excinfo.value)

