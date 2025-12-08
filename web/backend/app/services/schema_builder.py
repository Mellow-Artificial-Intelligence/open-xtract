"""Dynamic Pydantic model generation from user-defined schemas."""

from typing import Any

from pydantic import BaseModel, Field, create_model

FIELD_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
}


def build_pydantic_model(schema_definition: dict, model_name: str = "DynamicSchema") -> type[BaseModel]:
    """
    Convert a JSON schema definition to a Pydantic model at runtime.

    Supports:
    - Basic types: string, integer, float, boolean
    - Arrays with typed items
    - Nested objects
    - Optional fields with defaults

    Args:
        schema_definition: Dict with 'fields' key containing field definitions
        model_name: Name for the generated model class

    Returns:
        A dynamically created Pydantic BaseModel class
    """
    fields: dict[str, Any] = {}

    for field_def in schema_definition.get("fields", []):
        field_name = field_def["name"]
        is_required = field_def.get("required", True)
        description = field_def.get("description", "")
        default = field_def.get("default")

        python_type = _resolve_type(field_def, field_name)

        if is_required:
            fields[field_name] = (python_type, Field(description=description))
        else:
            if default is not None:
                fields[field_name] = (python_type | None, Field(default=default, description=description))
            else:
                fields[field_name] = (python_type | None, Field(default=None, description=description))

    # Clean model name (remove spaces, special chars)
    clean_name = "".join(c for c in model_name if c.isalnum())
    return create_model(clean_name or "DynamicSchema", **fields)


def _resolve_type(field_def: dict, parent_name: str = "") -> type:
    """Resolve the Python type from a field definition."""
    field_type = field_def["type"]

    if field_type == "array":
        items_def = field_def.get("items", {})
        if items_def:
            if items_def.get("type") == "object":
                # Nested object in array
                nested_fields = items_def.get("fields", [])
                nested_model = build_pydantic_model(
                    {"fields": nested_fields},
                    model_name=f"{parent_name}Item"
                )
                return list[nested_model]
            else:
                item_type = FIELD_TYPE_MAP.get(items_def.get("type", "string"), str)
                return list[item_type]
        return list[Any]

    elif field_type == "object":
        nested_fields = field_def.get("fields", [])
        if nested_fields:
            return build_pydantic_model(
                {"fields": nested_fields},
                model_name=f"{parent_name}Nested"
            )
        return dict[str, Any]

    return FIELD_TYPE_MAP.get(field_type, str)


def validate_schema_definition(schema_definition: dict) -> list[str]:
    """
    Validate a schema definition and return a list of errors.

    Returns empty list if valid.
    """
    errors = []

    if "fields" not in schema_definition:
        errors.append("Schema must have 'fields' key")
        return errors

    fields = schema_definition["fields"]
    if not isinstance(fields, list):
        errors.append("'fields' must be a list")
        return errors

    if len(fields) == 0:
        errors.append("Schema must have at least one field")
        return errors

    field_names = set()
    for i, field in enumerate(fields):
        if not isinstance(field, dict):
            errors.append(f"Field at index {i} must be an object")
            continue

        if "name" not in field:
            errors.append(f"Field at index {i} is missing 'name'")
        elif field["name"] in field_names:
            errors.append(f"Duplicate field name: {field['name']}")
        else:
            field_names.add(field["name"])

        if "type" not in field:
            errors.append(f"Field at index {i} is missing 'type'")
        elif field["type"] not in ["string", "integer", "float", "boolean", "array", "object"]:
            errors.append(f"Invalid type '{field['type']}' for field '{field.get('name', i)}'")

        # Validate nested structures
        if field.get("type") == "array" and "items" in field:
            items = field["items"]
            if items.get("type") == "object" and "fields" in items:
                nested_errors = validate_schema_definition({"fields": items["fields"]})
                errors.extend([f"In array '{field.get('name')}': {e}" for e in nested_errors])

        if field.get("type") == "object" and "fields" in field:
            nested_errors = validate_schema_definition({"fields": field["fields"]})
            errors.extend([f"In object '{field.get('name')}': {e}" for e in nested_errors])

    return errors


def schema_to_pydantic_code(schema_definition: dict, model_name: str = "ExtractedData") -> str:
    """
    Generate Python code for the Pydantic model.

    Useful for showing users what the generated model looks like.
    """
    lines = [
        "from typing import Optional",
        "from pydantic import BaseModel",
        "",
        "",
    ]

    # Collect nested models first
    nested_models = []
    _collect_nested_models(schema_definition.get("fields", []), nested_models)

    for nested_name, nested_fields in nested_models:
        lines.append(f"class {nested_name}(BaseModel):")
        for field in nested_fields:
            lines.append(_field_to_code(field, indent=4))
        lines.append("")

    # Main model
    lines.append(f"class {model_name}(BaseModel):")
    for field in schema_definition.get("fields", []):
        lines.append(_field_to_code(field, indent=4))

    return "\n".join(lines)


def _collect_nested_models(fields: list, result: list, prefix: str = "") -> None:
    """Recursively collect nested model definitions."""
    for field in fields:
        field_name = field.get("name", "")
        if field.get("type") == "object" and field.get("fields"):
            model_name = f"{prefix}{field_name.title()}Data"
            result.append((model_name, field["fields"]))
            _collect_nested_models(field["fields"], result, model_name)
        elif field.get("type") == "array" and field.get("items", {}).get("type") == "object":
            item_fields = field.get("items", {}).get("fields", [])
            if item_fields:
                model_name = f"{prefix}{field_name.title()}Item"
                result.append((model_name, item_fields))
                _collect_nested_models(item_fields, result, model_name)


def _field_to_code(field: dict, indent: int = 4) -> str:
    """Convert a field definition to Python code."""
    name = field["name"]
    ftype = field["type"]
    required = field.get("required", True)
    description = field.get("description", "")

    type_str = FIELD_TYPE_MAP.get(ftype, "str").__name__ if ftype in FIELD_TYPE_MAP else "str"

    if ftype == "array":
        items = field.get("items", {})
        if items.get("type") == "object":
            type_str = f"list[{name.title()}Item]"
        else:
            item_type = FIELD_TYPE_MAP.get(items.get("type", "string"), str).__name__
            type_str = f"list[{item_type}]"
    elif ftype == "object":
        type_str = f"{name.title()}Data"

    if not required:
        type_str = f"Optional[{type_str}]"

    prefix = " " * indent
    if description:
        return f'{prefix}{name}: {type_str}  # {description}'
    return f"{prefix}{name}: {type_str}"
