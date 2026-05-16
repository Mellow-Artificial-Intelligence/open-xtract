"""Tests for openextract._extract."""

from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from openextract import (
    ExtractionError,
    ModelError,
    SchemaValidationError,
    UrlFetchError,
    extract,
)
from openextract._extract import _get_media, _get_media_type

# ---------------------------------------------------------------------------
# _get_media_type
# ---------------------------------------------------------------------------


class TestGetMediaType:
    def test_returns_png_mime_type(self):
        assert _get_media_type("image.png") == "image/png"

    def test_returns_jpeg_mime_type(self):
        assert _get_media_type("photo.jpg") == "image/jpeg"

    def test_returns_text_mime_type(self):
        assert _get_media_type("notes.txt") == "text/plain"

    def test_returns_octet_stream_for_unknown_extension(self):
        assert _get_media_type("mystery.xyz123") == "application/octet-stream"

    def test_returns_octet_stream_for_no_extension(self):
        assert _get_media_type("README") == "application/octet-stream"

    def test_handles_full_path(self):
        assert _get_media_type("/tmp/nested/dir/file.png") == "image/png"


# ---------------------------------------------------------------------------
# _get_media
# ---------------------------------------------------------------------------


class TestGetMedia:
    def test_reads_local_file_with_known_extension(self, tmp_path):
        local = tmp_path / "hello.txt"
        local.write_bytes(b"hello world")

        media_bytes, media_type = _get_media(str(local))

        assert media_bytes == b"hello world"
        assert media_type == "text/plain"

    def test_reads_local_file_with_unknown_extension(self, tmp_path):
        odd_file = tmp_path / "data.weirdext"
        odd_file.write_bytes(b"opaque-bytes")

        media_bytes, media_type = _get_media(str(odd_file))

        assert media_bytes == b"opaque-bytes"
        assert media_type == "application/octet-stream"

    def test_reads_project_pdf_fixture(self):
        media_bytes, media_type = _get_media("tests/test.pdf")

        assert media_bytes.startswith(b"%PDF")
        assert media_type == "application/pdf"

    def test_missing_local_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            _get_media(str(missing))

    def test_fetches_https_url(self, mocker):
        fake_response = MagicMock()
        fake_response.content = b"<html>remote</html>"
        mock_get = mocker.patch("openextract._extract.httpx.get", return_value=fake_response)

        media_bytes, media_type = _get_media("https://example.com/page.html")

        mock_get.assert_called_once_with("https://example.com/page.html")
        assert media_bytes == b"<html>remote</html>"
        assert media_type == "text/html"

    def test_fetches_https_url_unknown_extension(self, mocker):
        fake_response = MagicMock()
        fake_response.content = b"raw-bytes"
        mocker.patch("openextract._extract.httpx.get", return_value=fake_response)

        media_bytes, media_type = _get_media("https://example.com/path/blob")

        assert media_bytes == b"raw-bytes"
        assert media_type == "application/octet-stream"


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


class _Person(BaseModel):
    name: str
    age: int


def _make_agent_mock(mocker, output=None, run_sync_side_effect=None):
    """Patch openextract._extract.Agent and return (AgentClass, instance, run_sync)."""
    agent_instance = MagicMock()
    if run_sync_side_effect is not None:
        agent_instance.run_sync.side_effect = run_sync_side_effect
    else:
        run_result = MagicMock()
        run_result.output = output
        agent_instance.run_sync.return_value = run_result
    agent_cls = mocker.patch("openextract._extract.Agent", return_value=agent_instance)
    return agent_cls, agent_instance


class TestExtract:
    def test_returns_schema_instance_from_local_file(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Ada", age=36)
        agent_cls, agent_instance = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            instructions="pull the person",
        )

        assert result is expected
        agent_cls.assert_called_once()
        # Non-ollama models pass the schema directly as output_type.
        kwargs = agent_cls.call_args.kwargs
        assert kwargs["instructions"] == "pull the person"
        assert kwargs["output_type"] is _Person
        agent_instance.run_sync.assert_called_once()

    def test_ollama_model_wraps_output_in_native_output(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Linus", age=54)
        agent_cls, _ = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="ollama:llama3",
            input_file=str(local),
        )

        assert result is expected
        output_type = agent_cls.call_args.kwargs["output_type"]
        # NativeOutput wraps the schema; assert it's not the bare schema class.
        assert output_type is not _Person
        assert type(output_type).__name__ == "NativeOutput"

    def test_http_status_error_is_wrapped(self, mocker):
        response = MagicMock()
        response.status_code = 503
        err = httpx.HTTPStatusError("boom", request=MagicMock(), response=response)
        mocker.patch("openextract._extract._get_media", side_effect=err)

        with pytest.raises(UrlFetchError, match="503"):
            extract(schema=_Person, model="openai:gpt-5", input_file="https://x/y")

    def test_request_error_is_wrapped(self, mocker):
        err = httpx.ConnectError("dns failure")
        mocker.patch("openextract._extract._get_media", side_effect=err)

        with pytest.raises(UrlFetchError, match="dns failure"):
            extract(schema=_Person, model="openai:gpt-5", input_file="https://x/y")

    def test_validation_error_is_wrapped(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        try:
            _Person(name="x", age="not-an-int")  # type: ignore[arg-type]
        except ValidationError as exc:
            validation_error = exc
        _make_agent_mock(mocker, run_sync_side_effect=validation_error)

        with pytest.raises(SchemaValidationError, match="Model output did not match schema"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_api_module_error_is_wrapped_as_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")

        class _ApiThing(Exception):
            pass

        # Force the exception's defining module to look like an "api" module.
        _ApiThing.__module__ = "some.vendor.api.client"
        _make_agent_mock(mocker, run_sync_side_effect=_ApiThing("upstream down"))

        with pytest.raises(ModelError, match="Model API error"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_message_mentioning_model_is_wrapped_as_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(
            mocker,
            run_sync_side_effect=RuntimeError("unknown model identifier"),
        )

        with pytest.raises(ModelError, match="Model API error"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_generic_exception_is_wrapped_as_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(mocker, run_sync_side_effect=RuntimeError("kaboom"))

        with pytest.raises(ExtractionError, match="Extraction failed: kaboom"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))
