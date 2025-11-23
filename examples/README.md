# OpenXtract Examples

This directory contains examples and cookbook recipes for using the OpenXtract library.

## Cookbook (`cookbook.py`)

The main cookbook file demonstrates basic usage patterns for OpenXtract:

- Setting up the OpenXtract client
- Defining Pydantic schemas for structured data extraction
- Extracting information from text using LLMs
- Configuration options (API keys, different models, etc.)

### Running the Cookbook

```bash
# From the project root
python examples/cookbook.py
```

### What it demonstrates:

1. **Basic Setup**: How to initialize OpenXtract with different model configurations
2. **Schema Definition**: Using Pydantic BaseModel to define extraction schemas
3. **Text Extraction**: Extracting structured data from plain text
4. **Field Descriptions**: Using Pydantic Field descriptions for better LLM guidance

## Adding New Examples

When adding new examples:

1. Create a new Python file in this directory
2. Follow the naming convention: `example_name.py`
3. Include docstrings and comments explaining what the example demonstrates
4. Add the example to this README.md file
