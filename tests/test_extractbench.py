"""Tests for scripts/extractbench.py (offline; no ExtractBench install required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extractbench  # noqa: E402


def test_pipeline_name_for_model_slugs_provider_prefix():
    assert extractbench.pipeline_name_for_model("openai:gpt-5") == "openextract_openai_gpt_5"
    assert extractbench.pipeline_name_for_model("xai:grok-4.3") == "openextract_xai_grok_4_3"
    assert extractbench.pipeline_name_for_model("openai:gpt-5", "custom") == "custom"


def test_add_additional_properties_false_closes_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"qty": {"type": "integer"}},
                },
            },
        },
        "$defs": {"Addr": {"type": "object", "properties": {"city": {"type": "string"}}}},
    }
    out = extractbench.add_additional_properties_false(schema)
    assert out["additionalProperties"] is False
    assert out["properties"]["lines"]["items"]["additionalProperties"] is False
    assert out["$defs"]["Addr"]["additionalProperties"] is False
    assert schema.get("additionalProperties") is None


def test_inline_json_schema_defs_resolves_local_refs():
    schema = {
        "type": "object",
        "properties": {"addr": {"$ref": "#/$defs/Addr"}},
        "$defs": {
            "Addr": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        },
    }
    out = extractbench.inline_json_schema_defs(schema)
    assert "$defs" not in out
    assert out["properties"]["addr"]["properties"]["city"]["type"] == "string"


def test_inline_json_schema_defs_leaves_recursive_refs():
    schema = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }
        },
    }
    out = extractbench.inline_json_schema_defs(schema)
    assert out["properties"]["child"]["properties"]["child"]["$ref"] == "#/$defs/Node"


def test_as_extracted_dict_from_passthrough_model():
    output = extractbench.ExtractedDocument.model_validate({"vendor": "Acme", "total": 1.5})
    assert extractbench.as_extracted_dict(output) == {"vendor": "Acme", "total": 1.5}


def test_as_extracted_dict_rejects_non_objects():
    with pytest.raises(TypeError, match="Expected dict"):
        extractbench.as_extracted_dict("not-json")


def test_extract_document_with_test_model():
    schema = {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "total": {"type": "number"},
        },
        "required": ["vendor", "total"],
    }
    model = TestModel(custom_output_args={"vendor": "Acme", "total": 12.5})
    data, usage = extractbench.extract_document(
        b"%PDF-fixture",
        schema,
        model,
        media_type="application/pdf",
        max_retries=0,
    )
    assert data["vendor"] == "Acme"
    assert data["total"] == 12.5
    assert usage.input_tokens >= 0


def test_parse_args_test_split_and_model():
    args = extractbench.parse_args(["--model", "openai:gpt-5", "--test"])
    assert args.model == "openai:gpt-5"
    assert args.test is True
    assert args.max_concurrent == 4


def test_parse_args_serve_without_name():
    args = extractbench.parse_args(["--serve"])
    assert args.serve == ""


def test_require_model_uses_env(monkeypatch):
    monkeypatch.setenv("OPENEXTRACT_MODEL", "xai:grok-4.3")
    args = extractbench.parse_args(["--test"])
    assert extractbench._require_model(args) == "xai:grok-4.3"


def test_require_model_exits_without_model(monkeypatch, capsys):
    monkeypatch.delenv("OPENEXTRACT_MODEL", raising=False)
    args = extractbench.parse_args(["--test"])
    with pytest.raises(SystemExit) as exc:
        extractbench._require_model(args)
    assert exc.value.code == 2
    assert "--model is required" in capsys.readouterr().err


def test_main_install_bootstraps_without_model(monkeypatch, capsys):
    monkeypatch.setattr(extractbench, "ensure_extract_bench", lambda: None)
    assert extractbench.main(["--install"]) == 0
    assert "ExtractBench ready" in capsys.readouterr().out
