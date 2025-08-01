"""LangGraph workflow definition for PDF processing."""

from typing import TypedDict
from functools import partial

from langgraph.graph import StateGraph, END

from ..models.base import BaseLLMProvider
from ..schemas.types import ProcessingState
from .nodes.pdf_processing import process_pdf_node
from .nodes.document_export import export_documents_node
from .nodes.intelligent_grep import intelligent_grep_node
from .nodes.structured_output import structured_output_node


def create_workflow(provider: BaseLLMProvider) -> StateGraph:
    """Create the LangGraph workflow for PDF processing.
    
    Args:
        provider: LLM provider instance
    
    Returns:
        Configured StateGraph workflow
    """
    # Create the graph
    workflow = StateGraph(ProcessingState)
    
    # Add nodes with provider where needed
    workflow.add_node(
        "process_pdf",
        partial(process_pdf_node, provider=provider)
    )
    workflow.add_node(
        "export_documents",
        export_documents_node
    )
    workflow.add_node(
        "intelligent_grep",
        partial(intelligent_grep_node, provider=provider)
    )
    workflow.add_node(
        "structured_output",
        structured_output_node
    )
    
    # Define the flow
    workflow.set_entry_point("process_pdf")
    
    # Add edges
    workflow.add_edge("process_pdf", "export_documents")
    workflow.add_edge("export_documents", "intelligent_grep")
    workflow.add_edge("intelligent_grep", "structured_output")
    workflow.add_edge("structured_output", END)
    
    # Compile the graph
    return workflow.compile()