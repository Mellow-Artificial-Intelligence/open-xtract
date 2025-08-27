from __future__ import annotations

from typing import Any

import pytest

from open_xtract import OpenXtract


class _FakeChat:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # capture constructor kwargs for assertions
        self.kwargs = kwargs


def test_create_llm_default_openai_base_url(monkeypatch: Any) -> None:
    # Patch ChatOpenAI import to our fake, and check base_url default
    import sys
    import types

    def fake_constructor(*args: Any, **kwargs: Any) -> _FakeChat:
        return _FakeChat(*args, **kwargs)

    fake_openai_mod = types.SimpleNamespace(ChatOpenAI=fake_constructor)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_openai_mod)

    # Instantiate OpenXtract which calls _create_llm -> ChatOpenAI with default base_url
    ox = OpenXtract(model="gpt-5-nano", api_key="KEY")
    assert isinstance(ox._llm, _FakeChat)
    assert ox._llm.kwargs.get("base_url") == "https://api.openai.com/v1"


def test_create_llm_import_errors(monkeypatch: Any) -> None:
    # Force import errors for providers and assert helpful messages
    def raise_import(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - function signature
        raise ImportError("missing")

    monkeypatch.setattr(OpenXtract, "_create_llm", OpenXtract._create_llm)
    
    # More reliable approach: temporarily remove provider modules from sys.modules and patch import
    import sys
    missing = object()
    sys.modules.pop("langchain_anthropic", None)
    sys.modules.pop("langchain_google_genai", None)

    # Patch to raise ImportError when attempting to import provider modules by name
    original_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name in ("langchain_anthropic", "langchain_google_genai"):
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr("builtins.__import__", fake_import)
        # Claude
        with pytest.raises(ImportError) as exc1:
            OpenXtract(model="claude-3")._create_llm("claude-3", None, None)
        assert "langchain-anthropic is required" in str(exc1.value)

        # Gemini
        with pytest.raises(ImportError) as exc2:
            OpenXtract(model="gemini-pro")._create_llm("gemini-pro", None, None)
        assert "langchain-google-genai is required" in str(exc2.value)

