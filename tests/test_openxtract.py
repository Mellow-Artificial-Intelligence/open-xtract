import io
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from open_xtract.exceptions import ConfigurationError, InputError, ProviderError
from open_xtract.main import OpenXtract
from PIL import Image
from pydantic import BaseModel

NUM_PARTS_TEXT_PLUS_TWO_IMAGES = 3
NUM_PARTS_TEXT_PLUS_THREE_IMAGES = 4


class MockSchema(BaseModel):
    """Mock Pydantic schema for testing."""

    name: str
    value: int


class TestOpenXtract:
    """Test cases for OpenXtract class."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key-12345"})
    def test_init(self):
        """Test OpenXtract initialization."""
        extractor = OpenXtract(model="openai:gpt-4")

        assert extractor._model_string == "openai:gpt-4"
        assert extractor._provider == "openai"
        assert extractor._model == "gpt-4"
        assert extractor._api_key == "test-api-key-12345"
        assert extractor._base_url == "https://api.openai.com/v1"
        assert extractor._llm is not None

    @patch("open_xtract.main.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key-12345"})
    def test_extract_with_mock(self, mock_chat_openai):
        """Test extract method with mocked LLM."""
        # Setup mock
        mock_llm = Mock()
        mock_response = MockSchema(name="test", value=42)
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        # Create extractor
        extractor = OpenXtract(model="openai:gpt-4")

        # Test extract method (text input)
        text_input = "name: test, value: 42"
        result = extractor.extract(text_input, MockSchema)

        # Verify the result
        assert result.name == "test"
        assert result.value == 42  # noqa: PLR2004

        # Verify the mock was called correctly
        mock_llm.with_structured_output.assert_called_once_with(MockSchema)
        mock_llm.with_structured_output.return_value.invoke.assert_called_once_with(text_input)

        # Verify ChatOpenAI was created with correct parameters
        mock_chat_openai.assert_called_once_with(
            model="gpt-4", base_url="https://api.openai.com/v1", api_key="test-api-key-12345"
        )

    @patch("open_xtract.main.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key-12345"})
    def test_extract_image_bytes_openai(self, mock_chat_openai):
        mock_llm = Mock()
        mock_response = MockSchema(name="img", value=1)
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        # Patch HumanMessage to capture content
        class FakeHumanMessage:
            def __init__(self, content):
                self.content = content

        with patch("open_xtract.main.HumanMessage", FakeHumanMessage):
            ox = OpenXtract(model="openai:gpt-4o-mini")
            # Generate a small PNG in memory
            buf = io.BytesIO()
            Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
            img_bytes = buf.getvalue()

            result = ox.extract(img_bytes, MockSchema)

        assert result.name == "img"
        assert result.value == 1  # noqa: PLR2004

        # Validate content structure
        invoke_args, _ = mock_llm.with_structured_output.return_value.invoke.call_args
        assert isinstance(invoke_args[0], list)
        human_msg = invoke_args[0][0]
        parts = human_msg.content
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @patch("open_xtract.main.ChatAnthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-api-key-12345"})
    def test_extract_image_bytes_anthropic(self, mock_chat_anthropic):
        mock_llm = Mock()
        mock_response = MockSchema(name="img", value=2)
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_response
        mock_chat_anthropic.return_value = mock_llm

        class FakeHumanMessage:
            def __init__(self, content):
                self.content = content

        with patch("open_xtract.main.HumanMessage", FakeHumanMessage):
            ox = OpenXtract(model="anthropic:claude-3-5-sonnet")
            buf = io.BytesIO()
            Image.new("RGB", (2, 2), color=(0, 255, 0)).save(buf, format="PNG")
            img_bytes = buf.getvalue()

            result = ox.extract(img_bytes, MockSchema)

        assert result.value == 2  # noqa: PLR2004
        invoke_args, _ = mock_llm.with_structured_output.return_value.invoke.call_args
        human_msg = invoke_args[0][0]
        parts = human_msg.content
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image"
        assert parts[1]["source"]["type"] == "base64"
        assert parts[1]["source"]["media_type"].startswith("image/")

    @patch("open_xtract.main.ChatOpenAI")
    @patch("open_xtract.main.importlib.import_module")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key-12345"})
    def test_extract_pdf_bytes_openai(self, mock_import_module, mock_chat_openai):
        mock_llm = Mock()
        mock_response = MockSchema(name="pdf", value=3)
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        # Fake fitz module
        class FakePixmap:
            def tobytes(self, fmt):
                return b"PNGDATA"

        class FakePage:
            def get_pixmap(self, dpi):
                return FakePixmap()

        class FakeDoc:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def __iter__(self):
                return iter([FakePage(), FakePage()])

        fake_fitz = SimpleNamespace(open=lambda stream, filetype: FakeDoc())
        mock_import_module.return_value = fake_fitz

        # Construct without __init__ to avoid creating real clients
        ox = object.__new__(OpenXtract)
        ox._provider = "openai"
        ox._llm = mock_llm
        pdf_bytes = b"%PDF- FAKE"
        result = ox.extract(pdf_bytes, MockSchema)

        assert result.value == 3  # noqa: PLR2004
        invoke_args, _ = mock_llm.with_structured_output.return_value.invoke.call_args
        human_msg = invoke_args[0][0]
        parts = human_msg.content
        # 1 text + 2 images
        assert len(parts) == NUM_PARTS_TEXT_PLUS_TWO_IMAGES
        assert parts[1]["type"] == "image_url"
        assert parts[2]["type"] == "image_url"

    @patch("open_xtract.main.ChatAnthropic")
    @patch("open_xtract.main.importlib.import_module")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-api-key-12345"})
    def test_extract_pdf_bytes_anthropic(self, mock_import_module, mock_chat_anthropic):
        mock_llm = Mock()
        mock_response = MockSchema(name="pdf", value=4)
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_response
        mock_chat_anthropic.return_value = mock_llm

        class FakePixmap:
            def tobytes(self, fmt):
                return b"PNGDATA"

        class FakePage:
            def get_pixmap(self, dpi):
                return FakePixmap()

        class FakeDoc:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def __iter__(self):
                return iter([FakePage(), FakePage(), FakePage()])

        fake_fitz = SimpleNamespace(open=lambda stream, filetype: FakeDoc())
        mock_import_module.return_value = fake_fitz

        ox = object.__new__(OpenXtract)
        ox._provider = "anthropic"
        ox._llm = mock_llm
        pdf_bytes = b"%PDF- FAKE"
        result = ox.extract(pdf_bytes, MockSchema)

        assert result.value == 4  # noqa: PLR2004
        invoke_args, _ = mock_llm.with_structured_output.return_value.invoke.call_args
        human_msg = invoke_args[0][0]
        parts = human_msg.content
        # 1 text + 3 images
        assert len(parts) == NUM_PARTS_TEXT_PLUS_THREE_IMAGES
        assert parts[1]["type"] == "image"
        assert parts[2]["type"] == "image"
        assert parts[3]["type"] == "image"

    @patch("open_xtract.main.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key-12345"})
    def test_extract_unsupported_bytes_raises(self, mock_chat_openai):
        mock_chat_openai.return_value = Mock()
        ox = OpenXtract(model="openai:gpt-4o-mini")
        with pytest.raises(InputError):
            ox.extract(b"not-an-image-or-pdf", MockSchema)

    @patch("open_xtract.main.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key-12345"})
    def test_extract_invalid_type_raises(self, mock_chat_openai):
        mock_chat_openai.return_value = Mock()
        ox = OpenXtract(model="openai:gpt-4o-mini")
        with pytest.raises(InputError):
            ox.extract(123, MockSchema)  # type: ignore[arg-type]

    def test_custom_exceptions_model_string_format(self):
        """Test ConfigurationError for invalid model string format."""
        with pytest.raises(ConfigurationError) as exc_info:
            OpenXtract(model="invalid-format")
        
        assert "Provider required when model string doesn't include provider" in str(exc_info.value)
        assert "Or provide provider parameter" in str(exc_info.value)

    def test_custom_exceptions_unknown_provider(self):
        """Test ConfigurationError for unknown provider."""
        with pytest.raises(ConfigurationError) as exc_info:
            OpenXtract(model="unknown:gpt-4")
        
        assert "Unknown provider 'unknown'" in str(exc_info.value)
        assert "Available providers:" in str(exc_info.value)

    def test_custom_exceptions_missing_api_key(self):
        """Test ProviderError for missing API key."""
        with patch.dict(os.environ, {}, clear=True):  # Clear all environment variables
            with pytest.raises(ProviderError) as exc_info:
                OpenXtract(model="openai:gpt-4")
            
            assert "API key not found for provider 'openai'" in str(exc_info.value)
            assert "OPENAI_API_KEY" in str(exc_info.value)
            assert exc_info.value.provider == "openai"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "short"})
    def test_custom_exceptions_short_api_key(self):
        """Test ProviderError for short API key."""
        with pytest.raises(ProviderError) as exc_info:
            OpenXtract(model="openai:gpt-4")
        
        assert "Invalid API key format for provider 'openai'" in str(exc_info.value)
        assert "Ensure the API key is complete" in str(exc_info.value)
        assert exc_info.value.provider == "openai"

    def test_error_message_suggestions(self):
        """Test that error messages include helpful suggestions."""
        with pytest.raises(ConfigurationError) as exc_info:
            OpenXtract(model=":")
        
        error_str = str(exc_info.value)
        assert "Suggestions:" in error_str
        assert "Provider cannot be empty" in error_str

    @patch("open_xtract.main.ChatOpenAI")
    def test_init_with_direct_api_key(self, mock_chat_openai):
        """Test OpenXtract initialization with direct API key."""
        with patch.dict(os.environ, {}, clear=True):  # Clear all environment variables
            extractor = OpenXtract(
                model="openai:gpt-4",
                api_key="test-direct-api-key-12345"
            )

            assert extractor._provider == "openai"
            assert extractor._model == "gpt-4"
            assert extractor._api_key == "test-direct-api-key-12345"
            assert extractor._base_url == "https://api.openai.com/v1"
            mock_chat_openai.assert_called_once_with(
                model="gpt-4",
                base_url="https://api.openai.com/v1",
                api_key="test-direct-api-key-12345"
            )

    @patch("open_xtract.main.ChatOpenAI")
    def test_init_with_direct_base_url(self, mock_chat_openai):
        """Test OpenXtract initialization with direct base URL."""
        extractor = OpenXtract(
            model="openai:gpt-4",
            api_key="test-api-key-12345",
            base_url="https://custom-proxy.com/v1"
        )

        assert extractor._base_url == "https://custom-proxy.com/v1"
        mock_chat_openai.assert_called_once_with(
            model="gpt-4",
            base_url="https://custom-proxy.com/v1",
            api_key="test-api-key-12345"
        )

    @patch("open_xtract.main.ChatOpenAI")
    def test_init_model_without_colon(self, mock_chat_openai):
        """Test OpenXtract initialization with model without colon and provider parameter."""
        extractor = OpenXtract(
            model="gpt-4",
            provider="openai",
            api_key="test-api-key-12345"
        )

        assert extractor._provider == "openai"
        assert extractor._model == "gpt-4"
        assert extractor._api_key == "test-api-key-12345"
        mock_chat_openai.assert_called_once_with(
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-api-key-12345"
        )

    def test_init_model_without_colon_no_provider(self):
        """Test that model without colon raises error when provider not provided."""
        with pytest.raises(ConfigurationError) as exc_info:
            OpenXtract(model="gpt-4")

        assert "Provider required when model string doesn't include provider" in str(exc_info.value)

    @patch("open_xtract.main.ChatOpenAI")
    def test_api_key_priority_over_env_var(self, mock_chat_openai):
        """Test that direct API key takes priority over environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key-12345"}):
            extractor = OpenXtract(
                model="openai:gpt-4",
                api_key="direct-api-key-12345"
            )

            assert extractor._api_key == "direct-api-key-12345"
            mock_chat_openai.assert_called_once_with(
                model="gpt-4",
                base_url="https://api.openai.com/v1",
                api_key="direct-api-key-12345"
            )

    @patch("open_xtract.main.ChatOpenAI")
    def test_base_url_priority_over_default(self, mock_chat_openai):
        """Test that direct base URL takes priority over provider map default."""
        extractor = OpenXtract(
            model="openai:gpt-4",
            api_key="test-api-key-12345",
            base_url="https://custom-url.com/v1"
        )

        assert extractor._base_url == "https://custom-url.com/v1"
        mock_chat_openai.assert_called_once_with(
            model="gpt-4",
            base_url="https://custom-url.com/v1",
            api_key="test-api-key-12345"
        )

    @patch("open_xtract.main.ChatAnthropic")
    def test_init_anthropic_with_direct_api_key(self, mock_chat_anthropic):
        """Test Anthropic initialization with direct API key."""
        with patch.dict(os.environ, {}, clear=True):  # Clear all environment variables
            extractor = OpenXtract(
                model="anthropic:claude-3-5-sonnet",
                api_key="test-anthropic-key-12345"
            )

            assert extractor._provider == "anthropic"
            assert extractor._model == "claude-3-5-sonnet"
            assert extractor._api_key == "test-anthropic-key-12345"
            mock_chat_anthropic.assert_called_once_with(
                model="claude-3-5-sonnet",
                api_key="test-anthropic-key-12345"
            )

    @patch("open_xtract.main.ChatOpenAI")
    def test_model_without_colon_when_api_key_and_base_url_provided(self, mock_chat_openai):
        """Test that model string can be used without colon when api_key and base_url are provided."""
        with patch.dict(os.environ, {}, clear=True):  # Clear all environment variables
            extractor = OpenXtract(
                model="gpt-4o",
                api_key="test-api-key-12345",
                base_url="https://api.openai.com/v1"
            )

            assert extractor._provider == "openai"
            assert extractor._model == "gpt-4o"
            assert extractor._api_key == "test-api-key-12345"
            assert extractor._base_url == "https://api.openai.com/v1"
            mock_chat_openai.assert_called_once_with(
                model="gpt-4o",
                base_url="https://api.openai.com/v1",
                api_key="test-api-key-12345"
            )
