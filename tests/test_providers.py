"""Tests for LLM providers."""

import pytest
from unittest.mock import Mock, patch

from open_xtract.models import LLMConfig, create_provider
from open_xtract.models.providers import OpenAICompatibleProvider, VISION_CAPABLE_MODELS


class TestOpenAICompatibleProvider:
    """Test cases for OpenAI-compatible provider."""
    
    def test_provider_creation(self):
        """Test provider creation."""
        config = LLMConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini"
        )
        
        provider = create_provider(config)
        
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.config == config
    
    def test_vision_detection_known_models(self):
        """Test vision capability detection for known models."""
        vision_models = ["gpt-4o", "gpt-4-vision-preview", "claude-3-opus"]
        
        for model in vision_models:
            config = LLMConfig(
                provider="openai",
                api_key="test-key",
                model=model
            )
            provider = OpenAICompatibleProvider(config)
            
            assert provider.supports_vision() is True
    
    def test_vision_detection_non_vision_models(self):
        """Test vision capability detection for non-vision models."""
        non_vision_models = ["gpt-3.5-turbo", "text-davinci-003", "claude-2"]
        
        for model in non_vision_models:
            config = LLMConfig(
                provider="openai",
                api_key="test-key",
                model=model
            )
            provider = OpenAICompatibleProvider(config)
            
            assert provider.supports_vision() is False
    
    def test_vision_detection_heuristic(self):
        """Test vision detection heuristic for unknown models."""
        config = LLMConfig(
            provider="custom",
            api_key="test-key",
            model="custom-vision-model"
        )
        provider = OpenAICompatibleProvider(config)
        
        assert provider.supports_vision() is True
    
    @patch("open_xtract.models.providers.ChatOpenAI")
    def test_get_llm(self, mock_chat_openai):
        """Test LLM instance creation."""
        config = LLMConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=1000
        )
        provider = OpenAICompatibleProvider(config)
        
        llm = provider.get_llm()
        
        mock_chat_openai.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key="test-key",
            max_tokens=1000
        )
    
    def test_custom_base_url(self):
        """Test custom base URL configuration."""
        config = LLMConfig(
            provider="custom",
            api_key="test-key",
            model="llama2",
            base_url="http://localhost:11434/v1"
        )
        provider = OpenAICompatibleProvider(config)
        
        assert provider.config.base_url == "http://localhost:11434/v1"