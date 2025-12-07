"""Tests for OpenXtract."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from open_xtract import ExtractionError, InputError, OpenXtract


class MockSchema(BaseModel):
    """Mock Pydantic schema for testing."""

    name: str
    value: int


class InvoiceSchema(BaseModel):
    """Invoice schema for testing."""

    invoice_number: str
    date: str
    total: float
    vendor: str


@dataclass
class FakeResultMessage:
    """Fake ResultMessage for testing."""

    is_error: bool = False
    result: str | None = None
    structured_output: dict | None = None
    duration_ms: int = 100
    duration_api_ms: int = 50
    num_turns: int = 1
    session_id: str = "test-session"
    total_cost_usd: float | None = 0.001
    usage: dict | None = None


def create_mock_result_message(result: dict | None = None, is_error: bool = False):
    """Create a fake ResultMessage."""
    return FakeResultMessage(
        is_error=is_error,
        result=None,
        structured_output=result,
    )


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    """Create a temporary text file."""
    file = tmp_path / "test.txt"
    file.write_text("name: test, value: 42")
    return file


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    """Create a temporary JSON file."""
    file = tmp_path / "test.json"
    file.write_text('{"name": "json_test", "value": 100}')
    return file


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """Create a temporary image file (minimal PNG)."""
    file = tmp_path / "test.png"
    # Minimal valid PNG (1x1 transparent pixel)
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,  # IEND chunk
        0x42, 0x60, 0x82,
    ])
    file.write_bytes(png_data)
    return file


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    """Create a temporary PDF file (minimal valid PDF)."""
    file = tmp_path / "test.pdf"
    # Minimal valid PDF
    pdf_content = b"""%PDF-1.0
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000052 00000 n
0000000101 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
168
%%EOF"""
    file.write_bytes(pdf_content)
    return file


class TestOpenXtractInit:
    """Tests for OpenXtract initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        ox = OpenXtract()
        assert ox._model is None
        assert ox._permission_mode == "acceptEdits"
        assert ox._system_prompt is None

    def test_init_with_model(self):
        """Test initialization with model."""
        ox = OpenXtract(model="claude-sonnet-4-5")
        assert ox._model == "claude-sonnet-4-5"

    def test_init_with_system_prompt(self):
        """Test initialization with system prompt."""
        ox = OpenXtract(system_prompt="Extract dates in ISO format.")
        assert ox._system_prompt == "Extract dates in ISO format."

    def test_init_with_all_options(self):
        """Test initialization with all options."""
        ox = OpenXtract(
            model="claude-opus-4-5",
            permission_mode="bypassPermissions",
            system_prompt="Be precise.",
        )
        assert ox._model == "claude-opus-4-5"
        assert ox._permission_mode == "bypassPermissions"
        assert ox._system_prompt == "Be precise."


class TestOpenXtractExtract:
    """Tests for OpenXtract.extract() method."""

    @pytest.mark.asyncio
    async def test_extract_file_not_found(self):
        """Test that InputError is raised for non-existent file."""
        ox = OpenXtract()
        with pytest.raises(InputError) as exc_info:
            await ox.extract("/nonexistent/file.txt", MockSchema)
        assert "File not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_text_file(self, text_file: Path):
        """Test extraction from text file."""
        mock_result = create_mock_result_message({"name": "test", "value": 42})

        async def mock_query(*args, **kwargs):
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            result = await ox.extract(text_file, MockSchema)

        assert result.data.name == "test"
        assert result.data.value == 42

    @pytest.mark.asyncio
    async def test_extract_json_file(self, json_file: Path):
        """Test extraction from JSON file."""
        mock_result = create_mock_result_message({"name": "json_test", "value": 100})

        async def mock_query(*args, **kwargs):
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            result = await ox.extract(json_file, MockSchema)

        assert result.data.name == "json_test"
        assert result.data.value == 100

    @pytest.mark.asyncio
    async def test_extract_image_file(self, image_file: Path):
        """Test extraction from image file."""
        mock_result = create_mock_result_message({"name": "from_image", "value": 99})

        async def mock_query(*args, **kwargs):
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            result = await ox.extract(image_file, MockSchema)

        assert result.data.name == "from_image"
        assert result.data.value == 99

    @pytest.mark.asyncio
    async def test_extract_pdf_file(self, pdf_file: Path):
        """Test extraction from PDF file."""
        mock_result = create_mock_result_message(
            {"invoice_number": "INV-001", "date": "2025-01-01", "total": 100.0, "vendor": "ACME"}
        )

        async def mock_query(*args, **kwargs):
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            result = await ox.extract(pdf_file, InvoiceSchema)

        assert result.data.invoice_number == "INV-001"
        assert result.data.date == "2025-01-01"
        assert result.data.total == 100.0
        assert result.data.vendor == "ACME"

    @pytest.mark.asyncio
    async def test_extract_with_model(self, text_file: Path):
        """Test extraction with specific model."""
        mock_result = create_mock_result_message({"name": "test", "value": 1})
        captured_options = {}

        async def mock_query(prompt, options):
            captured_options.update(vars(options))
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract(model="claude-opus-4-5")
            await ox.extract(text_file, MockSchema)

        assert captured_options.get("model") == "claude-opus-4-5"

    @pytest.mark.asyncio
    async def test_extract_with_system_prompt(self, text_file: Path):
        """Test extraction with system prompt."""
        mock_result = create_mock_result_message({"name": "test", "value": 1})
        captured_options = {}

        async def mock_query(prompt, options):
            captured_options.update(vars(options))
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract(system_prompt="Extract carefully.")
            await ox.extract(text_file, MockSchema)

        assert captured_options.get("system_prompt") == "Extract carefully."

    @pytest.mark.asyncio
    async def test_extract_uses_read_tool(self, text_file: Path):
        """Test that extraction uses the Read tool."""
        mock_result = create_mock_result_message({"name": "test", "value": 1})
        captured_options = {}

        async def mock_query(prompt, options):
            captured_options.update(vars(options))
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            await ox.extract(text_file, MockSchema)

        assert "Read" in captured_options.get("allowed_tools", [])

    @pytest.mark.asyncio
    async def test_extract_uses_output_format(self, text_file: Path):
        """Test that extraction uses output_format with JSON schema."""
        mock_result = create_mock_result_message({"name": "test", "value": 1})
        captured_options = {}

        async def mock_query(prompt, options):
            captured_options.update(vars(options))
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            await ox.extract(text_file, MockSchema)

        output_format = captured_options.get("output_format")
        assert output_format is not None
        assert output_format["type"] == "json_schema"
        assert "properties" in output_format["schema"]


class TestOpenXtractErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_extraction_error_on_failure(self, text_file: Path):
        """Test ExtractionError when extraction fails."""
        mock_result = FakeResultMessage(is_error=True, result="Something went wrong")

        async def mock_query(*args, **kwargs):
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            with pytest.raises(ExtractionError) as exc_info:
                await ox.extract(text_file, MockSchema)
            assert "Extraction failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extraction_error_no_result(self, text_file: Path):
        """Test ExtractionError when no result returned."""

        async def mock_query(*args, **kwargs):
            # Yield nothing - empty async generator
            return
            yield  # noqa: B901 - makes this an async generator

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            with pytest.raises(ExtractionError) as exc_info:
                await ox.extract(text_file, MockSchema)
            assert "No result returned" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extraction_error_schema_mismatch(self, text_file: Path):
        """Test ExtractionError when result doesn't match schema."""
        mock_result = create_mock_result_message({"wrong_field": "value"})

        async def mock_query(*args, **kwargs):
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            with pytest.raises(ExtractionError) as exc_info:
                await ox.extract(text_file, MockSchema)
            assert "doesn't match schema" in str(exc_info.value)


class TestOpenXtractFileTypes:
    """Tests for different file type support."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename,content",
        [
            ("test.txt", "plain text content"),
            ("test.md", "# Markdown content"),
            ("test.csv", "name,value\ntest,42"),
            ("test.xml", "<root><name>test</name></root>"),
            ("test.html", "<html><body>test</body></html>"),
            ("test.yaml", "name: test\nvalue: 42"),
        ],
    )
    async def test_text_file_extensions(self, tmp_path: Path, filename: str, content: str):
        """Test extraction from various text file types."""
        file = tmp_path / filename
        file.write_text(content)

        mock_result = create_mock_result_message({"name": "test", "value": 42})

        async def mock_query(*args, **kwargs):
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            result = await ox.extract(file, MockSchema)

        assert result.data.name == "test"
        assert result.data.value == 42

    @pytest.mark.asyncio
    @pytest.mark.parametrize("extension", [".png", ".jpg", ".jpeg", ".gif", ".webp"])
    async def test_image_file_extensions(self, tmp_path: Path, extension: str):
        """Test extraction from various image file types."""
        file = tmp_path / f"test{extension}"
        # Write minimal binary content (not valid images, but enough for path testing)
        file.write_bytes(b"\x00" * 10)

        mock_result = create_mock_result_message({"name": "image", "value": 1})

        async def mock_query(*args, **kwargs):
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            result = await ox.extract(file, MockSchema)

        assert result.data.name == "image"

    @pytest.mark.asyncio
    async def test_pdf_file_extension(self, pdf_file: Path):
        """Test extraction from PDF file."""
        mock_result = create_mock_result_message({"name": "pdf", "value": 1})

        async def mock_query(*args, **kwargs):
            yield mock_result

        with (
            patch("open_xtract.main.query", mock_query),
            patch("open_xtract.main.ResultMessage", FakeResultMessage),
        ):
            ox = OpenXtract()
            result = await ox.extract(pdf_file, MockSchema)

        assert result.data.name == "pdf"
