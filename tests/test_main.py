from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai import AudioUrl, DocumentUrl, ImageUrl, VideoUrl

from main import (
    ExtractionError,
    SchemaValidationError,
    UrlFetchError,
    _get_media_url,
    extract,
)


class TestGetMediaUrl:
    @pytest.mark.parametrize(
        "url,expected_type",
        [
            ("https://example.com/doc.pdf", DocumentUrl),
            ("https://example.com/doc.docx", DocumentUrl),
            ("https://example.com/data.csv", DocumentUrl),
            ("https://example.com/file.txt", DocumentUrl),
        ],
    )
    def test_document_urls(self, url, expected_type):
        result = _get_media_url(url)
        assert isinstance(result, expected_type)
        assert result.url == url

    @pytest.mark.parametrize(
        "url,expected_type",
        [
            ("https://example.com/image.jpg", ImageUrl),
            ("https://example.com/image.jpeg", ImageUrl),
            ("https://example.com/image.png", ImageUrl),
            ("https://example.com/image.gif", ImageUrl),
            ("https://example.com/image.webp", ImageUrl),
        ],
    )
    def test_image_urls(self, url, expected_type):
        result = _get_media_url(url)
        assert isinstance(result, expected_type)
        assert result.url == url

    @pytest.mark.parametrize(
        "url,expected_type",
        [
            ("https://example.com/audio.mp3", AudioUrl),
            ("https://example.com/audio.wav", AudioUrl),
            ("https://example.com/audio.ogg", AudioUrl),
            ("https://example.com/audio.flac", AudioUrl),
        ],
    )
    def test_audio_urls(self, url, expected_type):
        result = _get_media_url(url)
        assert isinstance(result, expected_type)
        assert result.url == url

    @pytest.mark.parametrize(
        "url,expected_type",
        [
            ("https://example.com/video.mp4", VideoUrl),
            ("https://example.com/video.mov", VideoUrl),
            ("https://example.com/video.avi", VideoUrl),
            ("https://example.com/video.webm", VideoUrl),
        ],
    )
    def test_video_urls(self, url, expected_type):
        result = _get_media_url(url)
        assert isinstance(result, expected_type)
        assert result.url == url

    def test_unknown_extension_defaults_to_document(self):
        result = _get_media_url("https://example.com/file.unknown")
        assert isinstance(result, DocumentUrl)

    def test_no_extension_defaults_to_document(self):
        result = _get_media_url("https://example.com/file")
        assert isinstance(result, DocumentUrl)

    def test_case_insensitive(self):
        result = _get_media_url("https://example.com/image.PNG")
        assert isinstance(result, ImageUrl)


class TestExtract:
    def test_extract_returns_schema_instance(self, mocker):
        class TestSchema(BaseModel):
            title: str
            count: int

        mock_output = TestSchema(title="Test", count=42)
        mock_result = MagicMock()
        mock_result.output = mock_output

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.return_value = mock_result

        mock_agent = mocker.patch("main.Agent", return_value=mock_agent_instance)

        result = extract(
            schema=TestSchema,
            model="test-model",
            url="https://example.com/doc.pdf",
            instructions="test instructions",
        )

        assert result == mock_output
        assert isinstance(result, TestSchema)
        mock_agent.assert_called_once_with(
            "test-model",
            instructions="test instructions",
            output_type=TestSchema,
        )

    def test_extract_uses_correct_media_url_type(self, mocker):
        class TestSchema(BaseModel):
            data: str

        mock_result = MagicMock()
        mock_result.output = TestSchema(data="test")

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.return_value = mock_result

        mocker.patch("main.Agent", return_value=mock_agent_instance)

        extract(
            schema=TestSchema,
            model="test-model",
            url="https://example.com/image.png",
            instructions="test",
        )

        call_args = mock_agent_instance.run_sync.call_args[0][0]
        assert isinstance(call_args[1], ImageUrl)


class TestExtractErrorHandling:
    def test_http_status_error_raises_url_fetch_error(self, mocker):
        class TestSchema(BaseModel):
            data: str

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_request = MagicMock()

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.side_effect = httpx.HTTPStatusError(
            "Not Found", request=mock_request, response=mock_response
        )

        mocker.patch("main.Agent", return_value=mock_agent_instance)

        with pytest.raises(UrlFetchError) as exc_info:
            extract(
                schema=TestSchema,
                model="test-model",
                url="https://example.com/doc.pdf",
                instructions="test",
            )

        assert "404" in str(exc_info.value)

    def test_request_error_raises_url_fetch_error(self, mocker):
        class TestSchema(BaseModel):
            data: str

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.side_effect = httpx.ConnectError("Connection refused")

        mocker.patch("main.Agent", return_value=mock_agent_instance)

        with pytest.raises(UrlFetchError) as exc_info:
            extract(
                schema=TestSchema,
                model="test-model",
                url="https://example.com/doc.pdf",
                instructions="test",
            )

        assert "Connection refused" in str(exc_info.value)

    def test_validation_error_raises_schema_validation_error(self, mocker):
        class TestSchema(BaseModel):
            data: str

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.side_effect = ValidationError.from_exception_data(
            "TestSchema",
            [{"type": "missing", "loc": ("data",), "input": {}}],
        )

        mocker.patch("main.Agent", return_value=mock_agent_instance)

        with pytest.raises(SchemaValidationError) as exc_info:
            extract(
                schema=TestSchema,
                model="test-model",
                url="https://example.com/doc.pdf",
                instructions="test",
            )

        assert "schema" in str(exc_info.value).lower()

    def test_unknown_error_raises_extraction_error(self, mocker):
        class TestSchema(BaseModel):
            data: str

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.side_effect = RuntimeError("Something unexpected")

        mocker.patch("main.Agent", return_value=mock_agent_instance)

        with pytest.raises(ExtractionError) as exc_info:
            extract(
                schema=TestSchema,
                model="test-model",
                url="https://example.com/doc.pdf",
                instructions="test",
            )

        assert "Something unexpected" in str(exc_info.value)

    def test_exception_chaining_preserves_original(self, mocker):
        class TestSchema(BaseModel):
            data: str

        original_error = httpx.ConnectError("Original error")
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_sync.side_effect = original_error

        mocker.patch("main.Agent", return_value=mock_agent_instance)

        with pytest.raises(UrlFetchError) as exc_info:
            extract(
                schema=TestSchema,
                model="test-model",
                url="https://example.com/doc.pdf",
                instructions="test",
            )

        assert exc_info.value.__cause__ is original_error
