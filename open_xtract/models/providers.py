"""LLM provider implementations."""

import os
from typing import Any, Dict, List, Optional, Set

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_community.chat_models import ChatOpenAI
from httpx import AsyncClient

from .base import BaseLLMProvider, LLMConfig


VISION_CAPABLE_MODELS: Set[str] = {
    "gpt-4-vision-preview",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "gemini-pro-vision",
    "llava",
}


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for OpenAI and OpenAI-compatible APIs."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._vision_capable: Optional[bool] = None
    
    def get_llm(self) -> BaseLanguageModel:
        """Return the LangChain LLM instance."""
        if self._llm is None:
            kwargs = {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "api_key": self.config.api_key,
            }
            
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            
            if self.config.max_tokens:
                kwargs["max_tokens"] = self.config.max_tokens
            
            kwargs.update(self.config.additional_params)
            
            self._llm = ChatOpenAI(**kwargs)
        
        return self._llm
    
    def supports_vision(self) -> bool:
        """Check if the model supports vision/multimodal inputs."""
        if self._vision_capable is not None:
            return self._vision_capable
        
        # Check known vision-capable models
        model_lower = self.config.model.lower()
        for vision_model in VISION_CAPABLE_MODELS:
            if vision_model in model_lower:
                self._vision_capable = True
                return True
        
        # For unknown models, try to detect capability
        # This is a simple heuristic; you might want to implement
        # a more sophisticated detection method
        if "vision" in model_lower or "multimodal" in model_lower:
            self._vision_capable = True
        else:
            self._vision_capable = False
        
        return self._vision_capable
    
    async def agenerate(self, messages: List[BaseMessage], **kwargs) -> str:
        """Async generation method."""
        llm = self.get_llm()
        response = await llm.ainvoke(messages, **kwargs)
        return response.content


def create_provider(config: LLMConfig) -> BaseLLMProvider:
    """Factory function to create appropriate provider based on config."""
    # For now, we'll use OpenAI-compatible provider for everything
    # You can extend this to support specific providers with custom logic
    return OpenAICompatibleProvider(config)