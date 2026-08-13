"""Tests for scripts/extractbench.py (offline; no ExtractBench install required)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from openextract import SchemaValidationError

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


def test_output_type_for_schema_raises_instead_of_degrading(capsys):
    schema = {"title": "InvoiceList", "type": "array", "items": {"type": "object"}}
    with pytest.raises(SchemaValidationError, match="InvoiceList"):
        extractbench._output_type_for_schema(schema, "openai:gpt-5")
    assert "InvoiceList" in capsys.readouterr().err


def test_extract_document_rejects_unconvertible_schema(capsys):
    model = TestModel(custom_output_args={"vendor": "Acme"})
    with pytest.raises(SchemaValidationError):
        extractbench.extract_document(
            b"%PDF-fixture",
            {"type": "array", "items": {"type": "string"}},
            model,
            media_type="application/pdf",
            max_retries=0,
        )
    assert "warning:" in capsys.readouterr().err


def _fake_venv(tmp_path: Path) -> tuple[Path, Path]:
    venv_dir = tmp_path / "venv"
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    return venv_dir, python


def _patch_bootstrap(monkeypatch, tmp_path: Path, venv_dir: Path) -> list[list[str]]:
    monkeypatch.setattr(extractbench, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(extractbench, "VENV_DIR", venv_dir)
    monkeypatch.setattr(extractbench, "extract_bench_available", lambda: False)
    commands: list[list[str]] = []
    monkeypatch.setattr(extractbench, "_run", commands.append)
    return commands


def test_inside_benchmark_venv_compares_sys_prefix(monkeypatch, tmp_path):
    """Membership uses sys.prefix, not resolved executables (uv symlinks alias)."""
    venv_dir = tmp_path / "venv"
    monkeypatch.setattr(extractbench, "VENV_DIR", venv_dir)
    monkeypatch.setattr(sys, "prefix", str(venv_dir))
    assert extractbench._inside_benchmark_venv() is True
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "other-venv"))
    assert extractbench._inside_benchmark_venv() is False


def test_ensure_extract_bench_raises_inside_broken_benchmark_venv(monkeypatch, tmp_path):
    venv_dir, _python = _fake_venv(tmp_path)
    commands = _patch_bootstrap(monkeypatch, tmp_path, venv_dir)
    monkeypatch.setattr(sys, "prefix", str(venv_dir))
    with pytest.raises(RuntimeError, match="not importable"):
        extractbench.ensure_extract_bench()
    assert commands == []


def test_ensure_extract_bench_skips_install_when_already_present(monkeypatch, tmp_path):
    venv_dir, python = _fake_venv(tmp_path)
    commands = _patch_bootstrap(monkeypatch, tmp_path, venv_dir)
    monkeypatch.setattr(extractbench, "_venv_has_extract_bench", lambda _python: True)
    monkeypatch.setattr(sys, "argv", ["extractbench.py"])
    execs: list[tuple] = []
    monkeypatch.setattr(os, "execv", lambda *call: execs.append(call))
    extractbench.ensure_extract_bench()
    assert commands == []
    assert execs == [(str(python), [str(python), str(SCRIPTS / "extractbench.py")])]


def test_ensure_extract_bench_installs_when_probe_fails(monkeypatch, tmp_path):
    venv_dir, python = _fake_venv(tmp_path)
    commands = _patch_bootstrap(monkeypatch, tmp_path, venv_dir)
    monkeypatch.setattr(extractbench, "_venv_has_extract_bench", lambda _python: False)
    extractbench.ensure_extract_bench(reexec=False)
    assert len(commands) == 1
    assert extractbench.EXTRACTBENCH_GIT in commands[0]
    assert str(python) in commands[0]


def test_venv_has_extract_bench_false_when_python_missing(tmp_path):
    assert extractbench._venv_has_extract_bench(tmp_path / "missing") is False


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
