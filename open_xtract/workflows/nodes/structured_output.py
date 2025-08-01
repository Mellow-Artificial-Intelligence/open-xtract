"""Structured output node for creating final results."""

import json
from typing import Any, Dict, Union

from pydantic import BaseModel, ValidationError

from ...schemas.types import ProcessingState


async def structured_output_node(state: ProcessingState) -> ProcessingState:
    """Create structured output from search results.
    
    Args:
        state: Current processing state
    
    Returns:
        Updated state with structured output
    """
    try:
        if not state.search_results:
            state.error = "No search results to structure"
            return state
        
        # Build output based on schema type
        if isinstance(state.schema, type) and issubclass(state.schema, BaseModel):
            # Pydantic model
            output = _build_pydantic_output(state.schema, state.search_results)
        else:
            # JSON schema
            output = _build_json_output(state.schema, state.search_results)
        
        state.structured_output = output
        return state
        
    except Exception as e:
        state.error = f"Error creating structured output: {str(e)}"
        return state


def _build_pydantic_output(
    schema: type[BaseModel],
    search_results: Dict[str, Any]
) -> BaseModel:
    """Build output for Pydantic model."""
    # Map search results to model fields
    field_values = {}
    
    for field_name, field_info in schema.model_fields.items():
        if field_name in search_results:
            value = search_results[field_name]
            
            # Handle None values for optional fields
            if value is None and not field_info.is_required():
                continue
            
            field_values[field_name] = value
    
    # Try to create the model instance
    try:
        return schema(**field_values)
    except ValidationError as e:
        # Try with partial data
        partial_values = {}
        for field_name, field_info in schema.model_fields.items():
            if field_name in field_values:
                partial_values[field_name] = field_values[field_name]
            elif field_info.is_required():
                # Provide default based on type
                field_type = field_info.annotation
                if field_type == str:
                    partial_values[field_name] = ""
                elif field_type == int:
                    partial_values[field_name] = 0
                elif field_type == float:
                    partial_values[field_name] = 0.0
                elif field_type == bool:
                    partial_values[field_name] = False
                else:
                    partial_values[field_name] = None
        
        return schema(**partial_values)


def _build_json_output(
    schema: Dict[str, Any],
    search_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Build output for JSON schema."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    
    output = {}
    
    for field_name, field_schema in properties.items():
        if field_name in search_results:
            value = search_results[field_name]
            
            # Type conversion based on schema
            field_type = field_schema.get("type", "string")
            
            if value is not None:
                if field_type == "integer":
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        value = 0 if field_name in required else None
                elif field_type == "number":
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = 0.0 if field_name in required else None
                elif field_type == "boolean":
                    if isinstance(value, str):
                        value = value.lower() in ["true", "yes", "1", "on"]
                    else:
                        value = bool(value)
                elif field_type == "array":
                    if not isinstance(value, list):
                        value = [value] if value else []
                elif field_type == "object":
                    if not isinstance(value, dict):
                        value = {"value": value} if value else {}
            
            output[field_name] = value
        elif field_name in required:
            # Provide default for required fields
            field_type = field_schema.get("type", "string")
            if field_type == "string":
                output[field_name] = ""
            elif field_type == "integer":
                output[field_name] = 0
            elif field_type == "number":
                output[field_name] = 0.0
            elif field_type == "boolean":
                output[field_name] = False
            elif field_type == "array":
                output[field_name] = []
            elif field_type == "object":
                output[field_name] = {}
            else:
                output[field_name] = None
    
    return output