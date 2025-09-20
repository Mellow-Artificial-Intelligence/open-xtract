"""Streamlit demo for extracting structured data from PDFs with OpenXtract.

This sample app lets you upload a PDF, design a Pydantic schema via the UI,
then send the document to Grok-4-Fast on OpenRouter for structured extraction.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Type

import streamlit as st
from open_xtract import OpenXtract
from pydantic import BaseModel, Field, create_model

# Supported field types the user can pick for the dynamic schema builder.
FIELD_OPTIONS: Dict[str, Type[Any]] = {
    "Text": str,
    "Integer": int,
    "Float": float,
    "Boolean": bool,
    "Date (as text)": str,
}

TYPE_HINT_MAP: Dict[str, str] = {
    "Text": "str",
    "Integer": "int",
    "Float": "float",
    "Boolean": "bool",
    "Date (as text)": "str",
}

DEFAULT_MODEL = "openrouter:x-ai/grok-4-fast:free"


def _init_session_state() -> None:
    if "fields" not in st.session_state:
        st.session_state.fields = [
            {
                "name": "entity_name",
                "type": "Text",
                "required": True,
                "description": "The primary subject or entity the PDF describes.",
            }
        ]
    if "model_name" not in st.session_state:
        st.session_state.model_name = "ExtractionResult"
    if "model_string" not in st.session_state:
        st.session_state.model_string = DEFAULT_MODEL
    if "last_run_state" not in st.session_state:
        st.session_state.last_run_state = None
        st.session_state.last_run_message = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def _reset_extraction_state() -> None:
    """Clear cached extraction results so the UI reflects the latest inputs."""

    st.session_state.last_result = None
    st.session_state.last_run_state = None
    st.session_state.last_run_message = ""


def _trigger_rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if rerun is not None:
        rerun()
    else:  # pragma: no cover - fallback for older Streamlit
        st.experimental_rerun()


def _normalize_field_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", name or "").strip()
    if not cleaned:
        return ""
    parts = [part.lower() for part in cleaned.split() if part]
    normalized = "_".join(parts)
    normalized = normalized.strip("_")
    return normalized


def _normalize_field_input(idx: int) -> None:
    key = f"field-name-{idx}"
    current = st.session_state.get(key, "")
    st.session_state[key] = _normalize_field_name(current)


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    return str(value)


def _sanitize_class_name(name: str) -> str:
    """Turn arbitrary user input into a safe Pydantic model class name."""

    cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", name or "").strip()
    if not cleaned:
        return "ExtractionResult"

    parts = [part for part in cleaned.split() if part]
    candidate = "".join(part[:1].upper() + part[1:] for part in parts)
    if not candidate:
        candidate = "ExtractionResult"

    if not candidate[0].isalpha():
        candidate = f"Model{candidate}"

    if not candidate.isidentifier():
        candidate = re.sub(r"[^0-9a-zA-Z_]", "", candidate) or "ExtractionResult"
        if not candidate[0].isalpha():
            candidate = f"Model{candidate}" if candidate else "ExtractionResult"
    return candidate


def build_schema(
    model_name: str, fields: List[Dict[str, Any]]
) -> Tuple[Optional[Type[BaseModel]], str]:
    """Create a dynamic Pydantic model from validated field definitions."""

    class_name = _sanitize_class_name(model_name)
    if not fields:
        return None, class_name

    field_definitions: Dict[str, Tuple[Any, Any]] = {}

    for field in fields:
        python_type = FIELD_OPTIONS.get(field["type"], str)
        description = field.get("description") or None
        required = bool(field.get("required", True))

        annotation: Any = python_type
        if not required:
            annotation = Optional[python_type]
            default = Field(default=None, description=description)
        else:
            default = Field(..., description=description)

        field_definitions[field["name"]] = (annotation, default)

    dynamic_model = create_model(class_name, **field_definitions)
    return dynamic_model, class_name


def _generate_python_snippet(
    class_name: str, fields: List[Dict[str, Any]], model_string: str
) -> str:
    """Produce sample Python code that mirrors the current configuration."""

    imports = ["from open_xtract import OpenXtract", "from pydantic import BaseModel, Field"]
    if any(not field.get("required", True) for field in fields):
        imports.append("from typing import Optional")

    lines: List[str] = imports + ["", f"class {class_name}(BaseModel):"]

    if not fields:
        lines.append("    pass")
    else:
        for field in fields:
            field_name = field["name"]
            type_string = TYPE_HINT_MAP.get(field["type"], "str")
            required = field.get("required", True)
            description = field.get("description") or ""

            annotation = type_string if required else f"Optional[{type_string}]"
            default_token = "..." if required else "None"

            field_args = [default_token]
            if description:
                field_args.append(f"description={json.dumps(description)}")

            joined_args = ", ".join(field_args)
            lines.append(f"    {field_name}: {annotation} = Field({joined_args})")

    lines += [
        "",
        f"ox = OpenXtract(model={json.dumps(model_string)})",
        "with open(\"/path/to/document.pdf\", \"rb\") as f:",
        "    pdf_bytes = f.read()",
        "",
        f"result = ox.extract(pdf_bytes, {class_name})",
        "print(result)",
    ]

    return "\n".join(lines)


def main() -> None:
    st.set_page_config(page_title="OpenXtract Grok-4-Fast Demo", page_icon="📄", layout="wide")
    _init_session_state()

    st.title("OpenXtract + Grok-4-Fast PDF Extractor")
    st.caption(
        "Upload a PDF, outline the structured output you need, and extract it with Grok-4-Fast via OpenRouter."
    )

    existing_api_key = os.environ.get("OPENROUTER_API_KEY") or ""

    with st.sidebar:
        st.header("Configuration")

        if existing_api_key:
            st.success("Using OPENROUTER_API_KEY from the environment.")
            if st.toggle(
                "Override API key for this session",
                value=False,
                help="Optional override without touching your shell.",
            ):
                override_key = st.text_input(
                    "Alternate OpenRouter API Key",
                    type="password",
                    placeholder="sk-or-...",
                    help="Stored only in memory for this Streamlit session.",
                )
                if override_key:
                    os.environ["OPENROUTER_API_KEY"] = override_key
                    _reset_extraction_state()
        else:
            api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                placeholder="sk-or-...",
                help="Stored only in memory for this session. Get a key at https://openrouter.ai/keys.",
            )
            if api_key:
                os.environ["OPENROUTER_API_KEY"] = api_key
                _reset_extraction_state()

        st.text_input(
            "Model",
            key="model_string",
            help="Provider:model identifier for OpenXtract (e.g. openrouter:x-ai/grok-4-fast:free).",
            on_change=_reset_extraction_state,
        )

        st.text_input(
            "Schema name",
            key="model_name",
            help="Used as the dynamic Pydantic model class name.",
            on_change=_reset_extraction_state,
        )

        st.markdown(
            "Need help? See the docs at [OpenXtract](https://github.com/Mellow-Artificial-Intelligence/open-xtract)."
        )

    has_api_key = bool(os.environ.get("OPENROUTER_API_KEY"))

    st.subheader("1 · PDF Document")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Large PDFs are supported, but provider limits still apply.",
    )

    if uploaded_pdf is not None:
        size_mb = uploaded_pdf.size / (1024 * 1024)
        st.caption(f"Selected: {uploaded_pdf.name} ({size_mb:.2f} MB)")
    else:
        st.caption("No file selected yet.")

    st.divider()

    st.subheader("2 · Schema Designer")
    st.caption("Add fields, set their types, and describe what the extractor should return.")

    snapshot_before_edit = [dict(field) for field in st.session_state.fields]
    fields: List[Dict[str, Any]] = st.session_state.fields
    field_type_options = list(FIELD_OPTIONS.keys())

    if st.button("Add field", type="primary"):
        fields.append({"name": "", "type": "Text", "required": True, "description": ""})
        _reset_extraction_state()

    for idx, field in enumerate(list(fields)):
        st.markdown(f"**Field {idx + 1}**")
        st.text_input(
            "Name",
            value=field["name"],
            key=f"field-name-{idx}",
            on_change=lambda idx=idx: _normalize_field_input(idx),
            placeholder="company_name",
        )
        st.selectbox(
            "Type",
            options=field_type_options,
            index=field_type_options.index(field["type"]) if field["type"] in FIELD_OPTIONS else 0,
            key=f"field-type-{idx}",
        )
        st.checkbox(
            "Required",
            value=field["required"],
            key=f"field-required-{idx}",
        )
        st.text_area(
            "Description",
            value=field["description"],
            key=f"field-desc-{idx}",
            placeholder="Short guidance for the model.",
            height=70,
        )
        if st.button("Remove", key=f"remove-field-{idx}", type="secondary"):
            fields.pop(idx)
            _reset_extraction_state()
            st.session_state.fields = fields
            _trigger_rerun()

        st.divider()

    updated_fields: List[Dict[str, Any]] = []
    for idx, _ in enumerate(fields):
        raw_name = str(st.session_state.get(f"field-name-{idx}", "")).strip()
        normalized_name = _normalize_field_name(raw_name)
        if raw_name and normalized_name and raw_name != normalized_name:
            st.session_state[f"field-name-{idx}"] = normalized_name
        field_type = st.session_state.get(f"field-type-{idx}", "Text")
        updated_fields.append(
            {
                "name": normalized_name,
                "type": field_type if field_type in FIELD_OPTIONS else "Text",
                "required": bool(st.session_state.get(f"field-required-{idx}", True)),
                "description": str(st.session_state.get(f"field-desc-{idx}", "")).strip(),
            }
        )

    changed = updated_fields != snapshot_before_edit
    st.session_state.fields = updated_fields
    if changed:
        _reset_extraction_state()

    valid_fields: List[Dict[str, Any]] = []
    invalid_names: List[str] = []

    for field in st.session_state.fields:
        name = field["name"]
        if not name:
            continue
        if not name.isidentifier():
            invalid_names.append(name)
            continue
        valid_fields.append(field)

    if invalid_names:
        st.warning(
            "Rename fields that are not valid Python identifiers: "
            + ", ".join(f"`{name}`" for name in invalid_names)
        )

    schema_model, resolved_class_name = build_schema(st.session_state.model_name, valid_fields)

    if schema_model is None:
        st.info("Add at least one named field to generate a schema preview.")
    else:
        with st.expander(f"Pydantic model · {resolved_class_name}", expanded=False):
            st.json(schema_model.model_json_schema())

    st.divider()

    st.subheader("3 · Extract structured data")
    ready_to_run = schema_model is not None and uploaded_pdf is not None and has_api_key

    if not has_api_key:
        st.warning("Add an OpenRouter API key in the sidebar to enable extraction.")
    elif uploaded_pdf is None:
        st.info("Upload a PDF to enable extraction.")
    elif schema_model is None:
        st.info("Define at least one schema field to enable extraction.")

    extract_button = st.button(
        "Run extraction",
        type="primary",
        disabled=not ready_to_run,
    )

    if extract_button and ready_to_run:
        _reset_extraction_state()

        pdf_bytes = uploaded_pdf.getvalue()
        if not pdf_bytes:
            st.session_state.last_run_state = "error"
            st.session_state.last_run_message = "The uploaded file appears to be empty."
        else:
            with st.status("Calling Grok-4-Fast via OpenRouter…", expanded=True) as status:
                try:
                    ox = OpenXtract(model=st.session_state.model_string)
                    result = ox.extract(pdf_bytes, schema_model)
                except Exception as exc:  # pragma: no cover - Streamlit feedback path
                    status.update(label="Extraction failed", state="error")
                    st.session_state.last_run_state = "error"
                    st.session_state.last_run_message = str(exc)
                    st.exception(exc)
                else:
                    status.update(label="Extraction complete", state="complete")
                    st.session_state.last_run_state = "success"
                    st.session_state.last_run_message = "Structured extraction complete."

                    if isinstance(result, BaseModel):
                        st.session_state.last_result = result.model_dump()
                    elif isinstance(result, (dict, list)):
                        st.session_state.last_result = result
                    else:
                        st.session_state.last_result = result

    if st.session_state.last_run_state == "error" and st.session_state.last_run_message:
        st.error(st.session_state.last_run_message)

    if st.session_state.last_run_state == "success" and st.session_state.last_result is not None:
        st.success(st.session_state.last_run_message)
        last_result = st.session_state.last_result
        if isinstance(last_result, dict):
            rows = [
                {"Field": key, "Value": _stringify(value)}
                for key, value in last_result.items()
            ]
            st.table(rows)
        elif isinstance(last_result, list):
            if last_result and all(isinstance(item, dict) for item in last_result):
                st.dataframe(last_result, use_container_width=True)
            else:
                st.write(last_result)
        else:
            st.write(last_result)

    st.divider()

    st.subheader("4 · Use it in Python")
    if not valid_fields:
        st.info("Define at least one valid field to generate starter code.")
    else:
        snippet = _generate_python_snippet(
            resolved_class_name,
            valid_fields,
            st.session_state.model_string,
        )
        st.code(snippet, language="python")

        st.caption(
            "Copy this snippet into your project to reproduce the extraction outside Streamlit."
        )


if __name__ == "__main__":
    main()
