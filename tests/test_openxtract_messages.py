from __future__ import annotations

from typing import Any

import open_xtract.main as ox_module
from open_xtract import OpenXtract
from pydantic import BaseModel


class _FakeStructured:
    def __init__(self, schema: type[BaseModel]) -> None:
        self.schema = schema

    def invoke(self, messages: list[Any]) -> BaseModel:
        _FakeChatOpenAI.last_invocations.append(("invoke", self.schema, messages))
        return self.schema.model_validate(
            {
                k: 0 if v.annotation in (int, float) else "ok"
                for k, v in self.schema.model_fields.items()
            }
        )


class _FakeChatOpenAI:
    last_invocations: list[tuple[str, type[BaseModel], list[Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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


def test_extract_image_file_message_contains_data_url(monkeypatch: Any, tmp_path: Any) -> None:
    _FakeChatOpenAI.last_invocations.clear()
    fake_llm = _FakeChatOpenAI()
    monkeypatch.setattr(OpenXtract, "_create_llm", lambda *args, **kwargs: fake_llm)
    img_path = tmp_path / "img.png"
    img_path.write_bytes(b"bytes")
    ox = OpenXtract()
    _ = ox.extract(str(img_path), _Schema)
    _, _, messages = _FakeChatOpenAI.last_invocations[-1]
    assert len(messages) == 1
    content = messages[0].content
    assert isinstance(content, list)
    # Should include a data URL for the image
    urls = [c.get("image_url", {}).get("url") for c in content if isinstance(c, dict)]
    assert any(isinstance(u, str) and u.startswith("data:image/jpeg;base64,") for u in urls)


def test_extract_pdf_prompt_includes_docs(monkeypatch: Any, tmp_path: Any) -> None:
    _FakeChatOpenAI.last_invocations.clear()
    fake_llm = _FakeChatOpenAI()
    monkeypatch.setattr(OpenXtract, "_create_llm", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(ox_module, "PyPDFLoader", _FakePDFLoader)
    monkeypatch.setattr(ox_module, "LLMImageBlobParser", _FakeLLMImageBlobParser)

    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_text("dummy pdf content")

    ox = OpenXtract()
    _ = ox.extract(str(pdf_file), _Schema)
    _, _, messages = _FakeChatOpenAI.last_invocations[-1]
    # Prompt should contain the docs list string with DOC1/DOC2
    content = messages[0].content
    assert isinstance(content, str)
    assert "DOC1" in content and "DOC2" in content

