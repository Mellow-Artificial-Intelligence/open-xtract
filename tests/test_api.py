from typing import Any, Dict

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_extract_missing_schema_returns_422():
    # Missing required field "schema"
    payload = {
        "content_url": "https://example.com/file.pdf",
        "model": "google/gemini-2.5-flash-lite",
        # "schema": { ... }  # omitted
    }
    resp = client.post("/extract", json=payload)
    assert resp.status_code == 422


def test_extract_happy_path(monkeypatch):
    # Mock Extract.extract to avoid external network calls
    from open_xtract.main import Extract

    def fake_extract(
        self,
        *,
        content_url: str,
        filename: str = "content.pdf",
        schema: Dict[str, Any] | None = None,
        prompt: str = "Analyze this content",
        use_base64: bool = False,
        transforms: list[str] | None = None,
    ):
        return {"ok": True, "content_url": content_url, "prompt": prompt, "schema": schema}

    monkeypatch.setattr(Extract, "extract", fake_extract)

    payload = {
        "content_url": "https://example.com/file.pdf",
        "model": "google/gemini-2.5-flash-lite",
        "schema": {
            "name": "document_summary",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"}
                },
                "required": ["title"],
                "additionalProperties": False
            }
        },
        "prompt": "Extract the title"
    }
    resp = client.post("/extract", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["content_url"] == payload["content_url"]

