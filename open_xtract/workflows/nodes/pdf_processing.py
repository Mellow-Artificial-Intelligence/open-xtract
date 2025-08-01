"""PDF processing node for extracting content from PDFs."""

import base64
import io
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader

from ...models.base import BaseLLMProvider
from ...schemas.types import ProcessingState


async def process_pdf_node(state: ProcessingState, provider: BaseLLMProvider) -> ProcessingState:
    """Extract content from PDF, using vision if available.
    
    Args:
        state: Current processing state
        provider: LLM provider instance
    
    Returns:
        Updated state with extracted documents
    """
    try:
        pdf_path = state.pdf_path
        documents: List[Document] = []
        
        if state.supports_vision:
            # Use multimodal approach for vision-capable models
            documents = await _process_with_vision(pdf_path, provider)
        else:
            # Use standard text extraction
            documents = _process_without_vision(pdf_path)
        
        state.documents = documents
        return state
        
    except Exception as e:
        state.error = f"Error processing PDF: {str(e)}"
        return state


async def _process_with_vision(pdf_path: Path, provider: BaseLLMProvider) -> List[Document]:
    """Process PDF using vision-capable model to extract images and text."""
    documents = []
    
    try:
        # First, use PyPDFLoader to get the basic structure
        loader = PyPDFLoader(str(pdf_path), mode="page")
        base_documents = loader.load()
        
        # Then enhance with vision if PyMuPDF is available
        try:
            import fitz  # PyMuPDF for image extraction
            
            # Open the PDF
            pdf_document = fitz.open(str(pdf_path))
            
            for idx, base_doc in enumerate(base_documents):
                if idx >= len(pdf_document):
                    break
                    
                page = pdf_document[idx]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better quality
                img_data = pix.tobytes("png")
                img_base64 = base64.b64encode(img_data).decode()
                
                # Use multimodal LLM to extract content
                messages = [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Extract all text content from this PDF page, preserving structure and formatting. Include any tables, lists, or special formatting. If there are images with text, extract that text as well."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    )
                ]
                
                enhanced_content = await provider.agenerate(messages)
                
                # Create enhanced document
                doc = Document(
                    page_content=enhanced_content,
                    metadata={
                        **base_doc.metadata,
                        "original_text": base_doc.page_content,
                        "extraction_method": "vision_enhanced"
                    }
                )
                documents.append(doc)
            
            pdf_document.close()
            
        except ImportError:
            # PyMuPDF not available, use base documents with OCR if possible
            print("PyMuPDF not available for vision extraction, using standard PyPDFLoader")
            for doc in base_documents:
                doc.metadata["extraction_method"] = "text_only"
            documents = base_documents
            
    except Exception as e:
        print(f"Error in vision processing: {e}")
        # Fallback to standard extraction
        return _process_without_vision(pdf_path)
    
    return documents


def _process_without_vision(pdf_path: Path) -> List[Document]:
    """Process PDF using standard text extraction."""
    # Use PyPDFLoader with page mode for better structure preservation
    loader = PyPDFLoader(str(pdf_path), mode="page")
    documents = loader.load()
    
    # Update metadata to indicate extraction method
    for doc in documents:
        doc.metadata["extraction_method"] = "text_only"
    
    return documents