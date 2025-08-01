"""Main PDF processor class."""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

from .models.base import LLMConfig
from .models.providers import create_provider
from .schemas.types import ExtractionResult, ProcessingState
from .workflows.graph import create_workflow


class PDFProcessor:
    """Main class for processing PDFs and extracting structured data."""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize the PDF processor.
        
        Args:
            llm_provider: Name of the LLM provider
            api_key: API key for the provider
            model: Model name/ID
            base_url: Custom API endpoint for OpenAI-compatible providers
            **kwargs: Additional provider-specific parameters
        """
        if not api_key:
            import os
            api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")
            if not api_key:
                raise ValueError(f"API key required. Set {llm_provider.upper()}_API_KEY environment variable or pass api_key parameter.")
        
        self.config = LLMConfig(
            provider=llm_provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            additional_params=kwargs
        )
        
        self.provider = create_provider(self.config)
        self.workflow = create_workflow(self.provider)
    
    async def aprocess_pdf(
        self,
        pdf_path: Union[str, Path],
        schema: Union[type[BaseModel], Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> ExtractionResult:
        """Process a PDF file asynchronously and extract structured data.
        
        Args:
            pdf_path: Path to the PDF file
            schema: Pydantic model or JSON schema for extraction
            output_dir: Directory to save markdown files (default: same as PDF)
        
        Returns:
            ExtractionResult with extracted data
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return ExtractionResult(
                success=False,
                error=f"PDF file not found: {pdf_path}"
            )
        
        if output_dir is None:
            output_dir = pdf_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        state = ProcessingState(
            pdf_path=pdf_path,
            schema=schema,
            supports_vision=self.provider.supports_vision()
        )
        
        try:
            # Run the workflow
            final_state = await self.workflow.ainvoke(
                state,
                config={"configurable": {"thread_id": str(pdf_path)}}
            )
            
            if final_state.error:
                return ExtractionResult(
                    success=False,
                    error=final_state.error,
                    markdown_files=final_state.markdown_files
                )
            
            return ExtractionResult(
                success=True,
                data=final_state.structured_output,
                markdown_files=final_state.markdown_files,
                metadata={
                    "pdf_path": str(pdf_path),
                    "supports_vision": final_state.supports_vision,
                    "num_pages": len(final_state.documents)
                }
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=str(e)
            )
    
    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        schema: Union[type[BaseModel], Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> ExtractionResult:
        """Process a PDF file and extract structured data (sync wrapper).
        
        Args:
            pdf_path: Path to the PDF file
            schema: Pydantic model or JSON schema for extraction
            output_dir: Directory to save markdown files (default: same as PDF)
        
        Returns:
            ExtractionResult with extracted data
        """
        return asyncio.run(self.aprocess_pdf(pdf_path, schema, output_dir))