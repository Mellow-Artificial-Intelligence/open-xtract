"""Command-line interface for open-xtract."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, create_model

from .processor import PDFProcessor


def cli_main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="open-xtract",
        description="Extract structured data from PDFs using LangGraph and LLMs"
    )
    
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument(
        "--schema", "-s",
        type=str,
        required=True,
        help="Path to JSON schema file or inline JSON schema"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Directory to save markdown files (default: same as PDF)"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="openai",
        help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="API key for the provider (or use environment variable)"
    )
    parser.add_argument(
        "--base-url", "-u",
        type=str,
        help="Custom API endpoint for OpenAI-compatible providers"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["json", "pretty"],
        default="pretty",
        help="Output format (default: pretty)"
    )
    
    args = parser.parse_args()
    
    # Load schema
    try:
        schema = _load_schema(args.schema)
    except Exception as e:
        print(f"Error loading schema: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create processor
    try:
        processor = PDFProcessor(
            llm_provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url
        )
    except Exception as e:
        print(f"Error initializing processor: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process PDF
    print(f"Processing {args.pdf_path}...")
    result = processor.process_pdf(
        pdf_path=args.pdf_path,
        schema=schema,
        output_dir=args.output_dir
    )
    
    if result.success:
        print(f"✓ Successfully extracted data")
        print(f"✓ Saved {len(result.markdown_files)} markdown files")
        
        if args.format == "json":
            # Output as JSON
            if isinstance(result.data, BaseModel):
                print(json.dumps(result.data.model_dump(), indent=2))
            else:
                print(json.dumps(result.data, indent=2))
        else:
            # Pretty print
            print("\nExtracted Data:")
            print("-" * 50)
            if isinstance(result.data, BaseModel):
                for field, value in result.data.model_dump().items():
                    print(f"{field}: {value}")
            else:
                for field, value in result.data.items():
                    print(f"{field}: {value}")
    else:
        print(f"✗ Extraction failed: {result.error}", file=sys.stderr)
        sys.exit(1)


def _load_schema(schema_arg: str) -> Any:
    """Load schema from file or string."""
    # Try to parse as JSON first
    try:
        schema_dict = json.loads(schema_arg)
        return schema_dict
    except json.JSONDecodeError:
        pass
    
    # Try to load as file
    schema_path = Path(schema_arg)
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            schema_dict = json.load(f)
        
        # Check if it's a Pydantic model definition
        if "model_name" in schema_dict and "fields" in schema_dict:
            # Convert to Pydantic model
            fields = {}
            for field_name, field_info in schema_dict["fields"].items():
                field_type = _get_python_type(field_info.get("type", "str"))
                fields[field_name] = (field_type, field_info.get("default", ...))
            
            return create_model(
                schema_dict.get("model_name", "DynamicModel"),
                **fields
            )
        else:
            # Regular JSON schema
            return schema_dict
    
    raise ValueError(f"Could not parse schema: {schema_arg}")


def _get_python_type(type_str: str) -> type:
    """Convert string type to Python type."""
    type_map = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": float,
        "bool": bool,
        "boolean": bool,
        "list": list,
        "array": list,
        "dict": dict,
        "object": dict
    }
    return type_map.get(type_str.lower(), str)


if __name__ == "__main__":
    cli_main()