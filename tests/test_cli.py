"""Tests for openextract._cli."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from openextract import (
    ExtractionError,
    ModelError,
    SchemaValidationError,
    UrlFetchError,
)
from openextract._cli import _resolve_schema, main


class _FixtureSchema(BaseModel):
    name: str
    age: int


class _NotASchema:
    """Plain class that is not a pydantic BaseModel subclass."""


def _patch_extract(mocker, return_value=None, side_effect=None):
    mock_extract = mocker.patch("openextract._cli.extract")
    if side_effect is not None:
        mock_extract.side_effect = side_effect
    else:
        mock_extract.return_value = return_value
    return mock_extract


# ---------------------------------------------------------------------------
# _resolve_schema
# ---------------------------------------------------------------------------


class TestResolveSchema:
    def test_resolves_valid_schema(self):
        cls = _resolve_schema("tests.test_cli:_FixtureSchema")
        assert cls is _FixtureSchema

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            _resolve_schema("tests.test_cli._FixtureSchema")

    def test_empty_module_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            _resolve_schema(":_FixtureSchema")

    def test_empty_class_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            _resolve_schema("tests.test_cli:")

    def test_missing_class_raises(self):
        with pytest.raises(ValueError, match="not found in module"):
            _resolve_schema("tests.test_cli:DoesNotExist")

    def test_non_basemodel_raises(self):
        with pytest.raises(ValueError, match="does not refer to a Pydantic BaseModel"):
            _resolve_schema("tests.test_cli:_NotASchema")

    def test_bad_module_raises_import_error(self):
        with pytest.raises(ImportError):
            _resolve_schema("definitely_not_a_real_module_xyz:Thing")


# ---------------------------------------------------------------------------
# argparse behavior
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_missing_required_args_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "usage" in captured.err.lower()

    def test_missing_schema_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["input.txt", "--model", "openai:gpt-5"])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "schema" in captured.err.lower()

    def test_missing_model_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["input.txt", "--schema", "tests.test_cli:_FixtureSchema"])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "model" in captured.err.lower()

    def test_invalid_output_choice_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "input.txt",
                    "--schema",
                    "tests.test_cli:_FixtureSchema",
                    "--model",
                    "openai:gpt-5",
                    "--output",
                    "yaml",
                ]
            )
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "output" in captured.err.lower()


# ---------------------------------------------------------------------------
# main success paths
# ---------------------------------------------------------------------------


class TestMainSuccess:
    def test_json_output_default(self, mocker, capsys):
        fake = MagicMock()
        fake.model_dump_json.return_value = '{\n  "name": "Ada",\n  "age": 36\n}'
        mock_extract = _patch_extract(mocker, return_value=fake)

        exit_code = main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "openai:gpt-5",
                "--instructions",
                "find the person",
            ]
        )

        assert exit_code == 0
        fake.model_dump_json.assert_called_once_with(indent=2)
        mock_extract.assert_called_once_with(
            schema=_FixtureSchema,
            model="openai:gpt-5",
            input_file="input.txt",
            instructions="find the person",
        )
        captured = capsys.readouterr()
        assert '"name": "Ada"' in captured.out

    def test_repr_output(self, mocker, capsys):
        fake = MagicMock()
        fake.__repr__ = lambda self: "_FixtureSchema(name='Linus', age=54)"
        _patch_extract(mocker, return_value=fake)

        exit_code = main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "openai:gpt-5",
                "--output",
                "repr",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "_FixtureSchema(name='Linus', age=54)" in captured.out

    def test_instructions_default_to_none(self, mocker):
        fake = MagicMock()
        fake.model_dump_json.return_value = "{}"
        mock_extract = _patch_extract(mocker, return_value=fake)

        exit_code = main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "openai:gpt-5",
            ]
        )

        assert exit_code == 0
        assert mock_extract.call_args.kwargs["instructions"] is None


# ---------------------------------------------------------------------------
# main error paths / exit codes
# ---------------------------------------------------------------------------


class TestMainErrorCodes:
    def _invoke(self, mocker, exc):
        _patch_extract(mocker, side_effect=exc)
        return main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "openai:gpt-5",
            ]
        )

    def test_url_fetch_error_returns_2(self, mocker, capsys):
        assert self._invoke(mocker, UrlFetchError("404")) == 2
        assert "404" in capsys.readouterr().err

    def test_schema_validation_error_returns_3(self, mocker, capsys):
        assert self._invoke(mocker, SchemaValidationError("bad shape")) == 3
        assert "bad shape" in capsys.readouterr().err

    def test_model_error_returns_4(self, mocker, capsys):
        assert self._invoke(mocker, ModelError("upstream")) == 4
        assert "upstream" in capsys.readouterr().err

    def test_extraction_error_returns_5(self, mocker, capsys):
        assert self._invoke(mocker, ExtractionError("misc")) == 5
        assert "misc" in capsys.readouterr().err

    def test_bad_schema_module_returns_1(self, capsys):
        exit_code = main(
            [
                "input.txt",
                "--schema",
                "definitely_not_a_real_module_xyz:Thing",
                "--model",
                "openai:gpt-5",
            ]
        )
        assert exit_code == 1
        assert "error" in capsys.readouterr().err.lower()

    def test_schema_not_basemodel_returns_1(self, capsys):
        exit_code = main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_NotASchema",
                "--model",
                "openai:gpt-5",
            ]
        )
        assert exit_code == 1
        assert "BaseModel" in capsys.readouterr().err
