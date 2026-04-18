"""Tests for openextract._extract."""

import pytest

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
