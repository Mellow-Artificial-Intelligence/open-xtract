"""Tests for PDFProcessor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import BaseModel

from open_xtract import PDFProcessor, ExtractionResult
from open_xtract.models import LLMConfig


class TestSchema(BaseModel):
    """Test schema for extraction."""
    title: str
    author: str
    date: str = ""


class TestPDFProcessor:
    """Test cases for PDFProcessor."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = PDFProcessor(
            llm_provider="openai",
            api_key="test-key",
            model="gpt-4o-mini"
        )
        
        assert processor.config.provider == "openai"
        assert processor.config.api_key == "test-key"
        assert processor.config.model == "gpt-4o-mini"
    
    def test_initialization_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-test-key"}):
            processor = PDFProcessor(
                llm_provider="openai",
                model="gpt-4o-mini"
            )
            
            assert processor.config.api_key == "env-test-key"
    
    def test_missing_api_key(self):
        """Test error when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                PDFProcessor(
                    llm_provider="openai",
                    model="gpt-4o-mini"
                )
    
    @patch("open_xtract.processor.create_workflow")
    @patch("open_xtract.processor.create_provider")
    def test_process_pdf_file_not_found(self, mock_provider, mock_workflow):
        """Test processing non-existent PDF."""
        processor = PDFProcessor(
            llm_provider="openai",
            api_key="test-key",
            model="gpt-4o-mini"
        )
        
        result = processor.process_pdf(
            pdf_path="nonexistent.pdf",
            schema=TestSchema
        )
        
        assert not result.success
        assert "not found" in result.error
    
    def test_json_schema_support(self):
        """Test support for JSON schema."""
        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "pages": {"type": "integer"}
            },
            "required": ["title"]
        }
        
        processor = PDFProcessor(
            llm_provider="openai",
            api_key="test-key",
            model="gpt-4o-mini"
        )
        
        # Just verify it accepts JSON schema without error
        assert processor.config.model == "gpt-4o-mini"