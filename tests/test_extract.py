from typing import Any, Dict

import pytest

from open_xtract.main import Extract, detect_content_kind


def test_detect_content_kind_heuristics(tmp_path):
    assert detect_content_kind("https://example.com/a.png") == "image"
    assert detect_content_kind("https://example.com/a.pdf") == "document"
    assert detect_content_kind("data:image/png;base64,xxx") == "image"
    f = tmp_path / "doc.pdf"
    f.write_text("ok")
    assert detect_content_kind(str(f)) == "document"


def test_extract_builds_payload_and_parses(monkeypatch):
    # Mock requests.post to return structured output
    class Resp:
        status_code = 200

        def json(self) -> Dict[str, Any]:
            return {
                "choices": [
                    {"message": {"content": '{"title": "Bitcoin", "main_points": ["P2P"]}'}}
                ]
            }

    def fake_post(url: str, headers: Dict[str, str], json: Dict[str, Any]):
        # ensure response_format and model present
        assert json["model"] == "google/gemini-2.5-flash-lite"
        assert "response_format" in json
        return Resp()

    monkeypatch.setattr("open_xtract.main.requests.post", fake_post)

    extractor = Extract(model="google/gemini-2.5-flash-lite")
    schema = {
        "name": "document_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "main_points": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["title", "main_points"],
            "additionalProperties": False
        }
    }
    res = extractor.extract(
        content_url="https://bitcoin.org/bitcoin.pdf",
        schema=schema,
        prompt="Extract"
    )
    assert isinstance(res, dict)
    assert res["title"] == "Bitcoin"

