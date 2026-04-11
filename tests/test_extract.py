"""Tests for openextract._extract."""

from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from openextract._extract import _get_media, _get_media_type, extract
from openextract.exceptions import (
    ExtractionError,
    ModelError,
    SchemaValidationError,
    UrlFetchError,
)


class SampleSchema(BaseModel):
    """Minimal schema used across extraction tests."""

    title: str
    page_count: int


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

    def test_fetches_https_url(self, mocker):
        mock_response = MagicMock()
        mock_response.content = b"remote-bytes"
        mock_get = mocker.patch("openextract._extract.httpx.get", return_value=mock_response)

        media_bytes, media_type = _get_media("https://example.com/image.png")

        mock_get.assert_called_once_with("https://example.com/image.png")
        assert media_bytes == b"remote-bytes"
        assert media_type == "image/png"

    def test_http_url_is_treated_as_local_path(self, mocker):
        """Only `https://` is routed through httpx; plain `http://` falls through to Path."""
        mock_get = mocker.patch("openextract._extract.httpx.get")

        with pytest.raises((FileNotFoundError, OSError)):
            _get_media("http://example.com/image.png")

        mock_get.assert_not_called()

    def test_missing_local_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            _get_media(str(missing))


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


class TestExtract:
    @pytest.fixture
    def patched_media(self, mocker):
        """Stub out filesystem / network reads inside extract()."""
        return mocker.patch(
            "openextract._extract._get_media",
            return_value=(b"fake-bytes", "image/png"),
        )

    @pytest.fixture
    def patched_agent_cls(self, mocker):
        """Stub out the pydantic-ai Agent constructor used by extract()."""
        return mocker.patch("openextract._extract.Agent")

    def test_returns_agent_output_on_success(self, patched_media, patched_agent_cls):
        expected = SampleSchema(title="Demo", page_count=3)
        agent_instance = MagicMock()
        agent_instance.run_sync.return_value = MagicMock(output=expected)
        patched_agent_cls.return_value = agent_instance

        result = extract(
            schema=SampleSchema,
            model="google-gla:gemini-3-flash-preview",
            input_file="https://example.com/image.png",
            instructions="Extract the title and page count.",
        )

        assert result is expected
        patched_media.assert_called_once_with(file_path="https://example.com/image.png")
        patched_agent_cls.assert_called_once_with(
            "google-gla:gemini-3-flash-preview",
            instructions="Extract the title and page count.",
            output_type=SampleSchema,
        )
        # Agent should receive the prompt string + a BinaryContent wrapping our bytes.
        (call_args,) = agent_instance.run_sync.call_args.args
        prompt, binary_content = call_args
        assert "Extract" in prompt
        assert binary_content.data == b"fake-bytes"
        assert binary_content.media_type == "image/png"

    def test_http_status_error_becomes_url_fetch_error(self, patched_agent_cls, mocker):
        request = httpx.Request("GET", "https://example.com/image.png")
        response = httpx.Response(status_code=404, request=request)
        mocker.patch(
            "openextract._extract._get_media",
            side_effect=httpx.HTTPStatusError("boom", request=request, response=response),
        )

        with pytest.raises(UrlFetchError, match="404"):
            extract(
                schema=SampleSchema,
                model="google-gla:gemini-3-flash-preview",
                input_file="https://example.com/image.png",
                instructions="x",
            )

    def test_request_error_becomes_url_fetch_error(self, patched_agent_cls, mocker):
        mocker.patch(
            "openextract._extract._get_media",
            side_effect=httpx.ConnectError("connection refused"),
        )

        with pytest.raises(UrlFetchError, match="connection refused"):
            extract(
                schema=SampleSchema,
                model="google-gla:gemini-3-flash-preview",
                input_file="https://example.com/image.png",
                instructions="x",
            )

    def test_validation_error_becomes_schema_validation_error(
        self, patched_media, patched_agent_cls
    ):
        try:
            SampleSchema(title="Demo", page_count="not-a-number")  # type: ignore[arg-type]
        except ValidationError as e:
            validation_error = e

        agent_instance = MagicMock()
        agent_instance.run_sync.side_effect = validation_error
        patched_agent_cls.return_value = agent_instance

        with pytest.raises(SchemaValidationError, match="did not match schema"):
            extract(
                schema=SampleSchema,
                model="google-gla:gemini-3-flash-preview",
                input_file="https://example.com/image.png",
                instructions="x",
            )

    def test_model_keyword_in_exception_becomes_model_error(self, patched_media, patched_agent_cls):
        agent_instance = MagicMock()
        agent_instance.run_sync.side_effect = RuntimeError("upstream model failure")
        patched_agent_cls.return_value = agent_instance

        with pytest.raises(ModelError, match="upstream model failure"):
            extract(
                schema=SampleSchema,
                model="google-gla:gemini-3-flash-preview",
                input_file="https://example.com/image.png",
                instructions="x",
            )

    def test_generic_exception_becomes_extraction_error(self, patched_media, patched_agent_cls):
        agent_instance = MagicMock()
        agent_instance.run_sync.side_effect = RuntimeError("something unexpected")
        patched_agent_cls.return_value = agent_instance

        with pytest.raises(ExtractionError, match="Extraction failed"):
            extract(
                schema=SampleSchema,
                model="google-gla:gemini-3-flash-preview",
                input_file="https://example.com/image.png",
                instructions="x",
            )
