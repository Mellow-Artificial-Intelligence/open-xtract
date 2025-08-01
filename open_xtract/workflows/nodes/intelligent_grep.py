"""Intelligent grep search node for schema-based extraction."""

import json
import re
from typing import Any, Dict, List, Set, Union

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ...models.base import BaseLLMProvider
from ...schemas.types import ProcessingState


async def intelligent_grep_node(state: ProcessingState, provider: BaseLLMProvider) -> ProcessingState:
    """Perform intelligent grep searches based on schema.
    
    Args:
        state: Current processing state
        provider: LLM provider instance
    
    Returns:
        Updated state with search results
    """
    try:
        if not state.documents:
            state.error = "No documents to search"
            return state
        
        # Get schema fields
        schema_fields = _extract_schema_fields(state.schema)
        
        # Generate initial search patterns
        search_patterns = await _generate_search_patterns(schema_fields, provider)
        
        # Perform searches
        search_results = {}
        for field, patterns in search_patterns.items():
            field_results = []
            for pattern in patterns:
                results = _search_documents(state.documents, pattern)
                field_results.extend(results)
            search_results[field] = field_results
        
        # Refine searches if needed
        refined_results = await _refine_searches(
            search_results,
            schema_fields,
            state.documents,
            provider
        )
        
        state.search_results = refined_results
        return state
        
    except Exception as e:
        state.error = f"Error in intelligent grep: {str(e)}"
        return state


def _extract_schema_fields(schema: Union[type[BaseModel], Dict[str, Any]]) -> Dict[str, Any]:
    """Extract field information from schema."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        # Pydantic model
        fields = {}
        for field_name, field_info in schema.model_fields.items():
            fields[field_name] = {
                "type": str(field_info.annotation),
                "description": field_info.description or field_name,
                "required": field_info.is_required()
            }
        return fields
    else:
        # JSON schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields = {}
        for field_name, field_info in properties.items():
            fields[field_name] = {
                "type": field_info.get("type", "string"),
                "description": field_info.get("description", field_name),
                "required": field_name in required
            }
        return fields


async def _generate_search_patterns(
    schema_fields: Dict[str, Any],
    provider: BaseLLMProvider
) -> Dict[str, List[str]]:
    """Generate grep patterns for each schema field."""
    patterns = {}
    
    for field_name, field_info in schema_fields.items():
        prompt = f"""Generate 3-5 regex patterns to find '{field_name}' in a document.
Field description: {field_info['description']}
Field type: {field_info['type']}

Consider common variations, formats, and labels for this field.
Return only the regex patterns, one per line."""
        
        messages = [
            SystemMessage(content="You are a regex expert helping to extract data from documents."),
            HumanMessage(content=prompt)
        ]
        
        response = await provider.agenerate(messages)
        
        # Parse patterns from response
        field_patterns = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Clean up the pattern
                pattern = line.strip('`').strip()
                field_patterns.append(pattern)
        
        patterns[field_name] = field_patterns[:5]  # Limit to 5 patterns
    
    return patterns


def _search_documents(documents: List[Any], pattern: str) -> List[Dict[str, Any]]:
    """Search documents using a regex pattern."""
    results = []
    
    try:
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error:
        # Invalid regex, skip
        return results
    
    for doc in documents:
        content = doc.page_content
        matches = regex.finditer(content)
        
        for match in matches:
            # Get context around the match
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 100)
            context = content[start:end]
            
            results.append({
                "page": doc.metadata.get("page", 0),
                "match": match.group(),
                "context": context,
                "pattern": pattern
            })
    
    return results


async def _refine_searches(
    initial_results: Dict[str, List[Dict[str, Any]]],
    schema_fields: Dict[str, Any],
    documents: List[Any],
    provider: BaseLLMProvider
) -> Dict[str, Any]:
    """Refine search results using LLM to extract exact values."""
    refined_results = {}
    
    for field_name, field_info in schema_fields.items():
        field_results = initial_results.get(field_name, [])
        
        if not field_results and field_info["required"]:
            # No results found, try alternative search
            field_results = await _alternative_search(
                field_name,
                field_info,
                documents,
                provider
            )
        
        if field_results:
            # Use LLM to extract the exact value
            value = await _extract_value(
                field_name,
                field_info,
                field_results,
                provider
            )
            refined_results[field_name] = value
        else:
            refined_results[field_name] = None
    
    return refined_results


async def _alternative_search(
    field_name: str,
    field_info: Dict[str, Any],
    documents: List[Any],
    provider: BaseLLMProvider
) -> List[Dict[str, Any]]:
    """Perform alternative search when initial patterns fail."""
    # Combine all document content
    full_text = "\n\n".join([doc.page_content for doc in documents[:3]])  # Limit to first 3 pages
    
    prompt = f"""Find the value for '{field_name}' in this text.
Field description: {field_info['description']}
Field type: {field_info['type']}

Text:
{full_text[:2000]}...

Return the exact text where you found this information."""
    
    messages = [
        SystemMessage(content="You are helping to extract specific information from documents."),
        HumanMessage(content=prompt)
    ]
    
    response = await provider.agenerate(messages)
    
    if response and response.strip():
        return [{
            "page": 0,
            "match": response.strip(),
            "context": response.strip(),
            "pattern": "llm_search"
        }]
    
    return []


async def _extract_value(
    field_name: str,
    field_info: Dict[str, Any],
    results: List[Dict[str, Any]],
    provider: BaseLLMProvider
) -> Any:
    """Extract the exact value from search results."""
    # Prepare context from results
    contexts = []
    for result in results[:5]:  # Limit to top 5 results
        contexts.append(f"Page {result['page']}: {result['context']}")
    
    context_text = "\n\n".join(contexts)
    
    prompt = f"""Extract the exact value for '{field_name}' from these search results.
Field description: {field_info['description']}
Field type: {field_info['type']}

Search results:
{context_text}

Return only the extracted value, no explanation."""
    
    messages = [
        SystemMessage(content="You are extracting specific values from document search results."),
        HumanMessage(content=prompt)
    ]
    
    response = await provider.agenerate(messages)
    
    # Parse response based on field type
    value = response.strip()
    
    if field_info["type"] in ["int", "integer", "<class 'int'>"]:
        try:
            return int(value)
        except ValueError:
            # Try to extract number from text
            numbers = re.findall(r'\d+', value)
            return int(numbers[0]) if numbers else None
    elif field_info["type"] in ["float", "number", "<class 'float'>"]:
        try:
            return float(value)
        except ValueError:
            # Try to extract number from text
            numbers = re.findall(r'\d+\.?\d*', value)
            return float(numbers[0]) if numbers else None
    elif field_info["type"] in ["bool", "boolean", "<class 'bool'>"]:
        return value.lower() in ["true", "yes", "1", "on"]
    else:
        return value