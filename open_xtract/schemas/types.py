"""Type definitions for the open-xtract library."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class ProcessingState(BaseModel):
    """State object for the LangGraph workflow."""
    
    pdf_path: Path
    documents: List[Document] = Field(default_factory=list)
    markdown_files: List[Path] = Field(default_factory=list)
    schema: Union[type[BaseModel], Dict[str, Any]]
    search_results: Dict[str, Any] = Field(default_factory=dict)
    structured_output: Optional[Any] = None
    error: Optional[str] = None
    supports_vision: bool = False


class ExtractionResult(BaseModel):
    """Result of the PDF extraction process."""
    
    success: bool
    data: Optional[Any] = None
    markdown_files: List[Path] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)