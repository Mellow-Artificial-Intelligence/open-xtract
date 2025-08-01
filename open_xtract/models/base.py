"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    
    provider: str = Field(..., description="Provider name (e.g., 'openai', 'anthropic')")
    api_key: str = Field(..., description="API key for the provider")
    model: str = Field(..., description="Model name/ID")
    base_url: Optional[str] = Field(None, description="Custom API endpoint for OpenAI-compatible providers")
    temperature: float = Field(0.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._llm: Optional[BaseLanguageModel] = None
    
    @abstractmethod
    def get_llm(self) -> BaseLanguageModel:
        """Return the LangChain LLM instance."""
        pass
    
    @abstractmethod
    def supports_vision(self) -> bool:
        """Check if the model supports vision/multimodal inputs."""
        pass
    
    @abstractmethod
    async def agenerate(self, messages: List[BaseMessage], **kwargs) -> str:
        """Async generation method."""
        pass
    
    def generate(self, messages: List[BaseMessage], **kwargs) -> str:
        """Sync generation method."""
        import asyncio
        return asyncio.run(self.agenerate(messages, **kwargs))