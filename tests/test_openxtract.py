import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from pydantic import BaseModel
from open_xtract.main import OpenXtract


class MockSchema(BaseModel):
    """Mock Pydantic schema for testing."""
    name: str
    value: int


class TestOpenXtract:
    """Test cases for OpenXtract class."""

    def test_init(self):
        """Test OpenXtract initialization."""
        extractor = OpenXtract(
            model="gpt-5-nano",
            base_url="https://api.openai.com/v1",
            api_key="test-key"
        )

        assert extractor._model_name == "gpt-5-nano"
        assert extractor._base_url == "https://api.openai.com/v1"
        assert extractor._api_key == "test-key"
        assert extractor._llm is not None

    def test_create_llm(self):
        """Test LLM creation."""
        extractor = OpenXtract(model="gpt-5-nano")
        llm = extractor._create_llm("gpt-5-nano", None, None)

        assert llm is not None
        # The actual LLM instance will be a ChatOpenAI object

    @patch('open_xtract.main.ChatOpenAI')
    def test_extract_with_mock(self, mock_chat_openai):
        """Test extract method with mocked LLM."""
        # Setup mock
        mock_llm = Mock()
        mock_response = MockSchema(name="test", value=42)
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        # Create extractor
        extractor = OpenXtract(model="gpt-5-nano")

        # Test extract method
        file_path = Path("/test/file.txt")
        result = extractor.extract(file_path, MockSchema)

        # Verify the result
        assert result.name == "test"
        assert result.value == 42

        # Verify the mock was called correctly
        mock_llm.with_structured_output.assert_called_once_with(MockSchema)
        mock_llm.with_structured_output.return_value.invoke.assert_called_once_with(file_path)
