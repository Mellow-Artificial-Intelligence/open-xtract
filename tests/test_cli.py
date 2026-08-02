"""Tests for openextract._cli."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from openextract import (
    ExtractionError,
    InputTooLargeError,
    ModelError,
    ProviderNotInstalledError,
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


def _patch_extract_with_usage(mocker, return_value=None, side_effect=None):
    mock_fn = mocker.patch("openextract._cli.extract_with_usage")
    if side_effect is not None:
        mock_fn.side_effect = side_effect
    else:
        mock_fn.return_value = return_value
    return mock_fn


def _patch_extract_many(mocker, return_value=None, side_effect=None):
    mock_fn = mocker.patch("openextract._cli.extract_many")
    if side_effect is not None:
        mock_fn.side_effect = side_effect
    else:
        mock_fn.return_value = return_value
    return mock_fn


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
            main(["input.txt", "--model", "xai:grok-4.3"])
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
                    "xai:grok-4.3",
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
    def test_max_input_bytes_is_forwarded(self, mocker, capsys):
        fake = _FixtureSchema(name="Ada", age=36)
        mock_extract = _patch_extract(mocker, return_value=fake)

        assert (
            main(
                [
                    "input.txt",
                    "--schema",
                    "tests.test_cli:_FixtureSchema",
                    "--model",
                    "openai:gpt-5",
                    "--max-input-bytes",
                    "1024",
                ]
            )
            == 0
        )

        assert mock_extract.call_args.kwargs["max_input_bytes"] == 1024
        capsys.readouterr()

    def test_loads_dotenv_at_application_boundary(self, mocker, capsys):
        load_dotenv = mocker.patch("openextract._cli.load_dotenv")
        _patch_extract(mocker, return_value=_FixtureSchema(name="Ada", age=36))

        assert (
            main(
                [
                    "input.txt",
                    "--schema",
                    "tests.test_cli:_FixtureSchema",
                    "--model",
                    "openai:gpt-5",
                ]
            )
            == 0
        )

        load_dotenv.assert_called_once_with()
        capsys.readouterr()

    def test_json_output_default(self, mocker, capsys):
        fake = _FixtureSchema(name="Ada", age=36)
        mock_extract = _patch_extract(mocker, return_value=fake)

        exit_code = main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--instructions",
                "find the person",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert '"name": "Ada"' in captured.out
        mock_extract.assert_called_once_with(
            schema=_FixtureSchema,
            model="xai:grok-4.3",
            input_file="input.txt",
            instructions="find the person",
            media_type=None,
            max_input_bytes=None,
            max_retries=0,
            retry_backoff=1.0,
            retry_max_backoff=60.0,
        )

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
                "xai:grok-4.3",
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
                "xai:grok-4.3",
            ]
        )

        assert exit_code == 0
        assert mock_extract.call_args.kwargs["instructions"] is None

    def test_retry_defaults(self, mocker):
        fake = MagicMock()
        fake.model_dump_json.return_value = "{}"
        mock_extract = _patch_extract(mocker, return_value=fake)

        main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
            ]
        )

        assert mock_extract.call_args.kwargs["max_retries"] == 0
        assert mock_extract.call_args.kwargs["retry_backoff"] == 1.0
        assert mock_extract.call_args.kwargs["retry_max_backoff"] == 60.0

    def test_retry_options_custom(self, mocker):
        fake = MagicMock()
        fake.model_dump_json.return_value = "{}"
        mock_extract = _patch_extract(mocker, return_value=fake)

        main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--max-retries",
                "3",
                "--retry-backoff",
                "2.5",
                "--retry-max-backoff",
                "12.0",
            ]
        )

        assert mock_extract.call_args.kwargs["max_retries"] == 3
        assert mock_extract.call_args.kwargs["retry_backoff"] == 2.5
        assert mock_extract.call_args.kwargs["retry_max_backoff"] == 12.0

    def test_invalid_max_retries_returns_1(self, capsys):
        exit_code = main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--max-retries",
                "-1",
            ]
        )

        assert exit_code == 1
        assert "max_retries" in capsys.readouterr().err

    def test_invalid_retry_backoff_returns_1(self, capsys):
        exit_code = main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--retry-backoff",
                "-0.1",
            ]
        )

        assert exit_code == 1
        assert "retry_backoff" in capsys.readouterr().err

    def test_invalid_retry_max_backoff_returns_1(self, capsys):
        exit_code = main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--retry-max-backoff",
                "-1",
            ]
        )

        assert exit_code == 1
        assert "retry_max_backoff" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main error paths / exit codes
# ---------------------------------------------------------------------------


class TestMainErrorCodes:
    def test_input_too_large_returns_5(self, mocker, capsys):
        _patch_extract(mocker, side_effect=InputTooLargeError("too large"))

        exit_code = main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "openai:gpt-5",
            ]
        )

        assert exit_code == 5
        assert "too large" in capsys.readouterr().err

    def _invoke(self, mocker, exc):
        _patch_extract(mocker, side_effect=exc)
        return main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
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

    def test_provider_not_installed_error_returns_6(self, mocker, capsys):
        exc = ProviderNotInstalledError("Install it with: pip install 'openextract[openai]'")
        assert self._invoke(mocker, exc) == 6
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "pip install 'openextract[openai]'" in captured.err

    def test_bad_schema_module_returns_1(self, capsys):
        exit_code = main(
            [
                "input.txt",
                "--schema",
                "definitely_not_a_real_module_xyz:Thing",
                "--model",
                "xai:grok-4.3",
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
                "xai:grok-4.3",
            ]
        )
        assert exit_code == 1
        assert "BaseModel" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# batch, usage, stdin
# ---------------------------------------------------------------------------


class TestMainBatchAndUsage:
    def test_multiple_files_use_extract_many(self, mocker, capsys):
        fake = MagicMock()
        fake.model_dump.return_value = {"name": "Ada", "age": 36}
        mock_many = _patch_extract_many(mocker, return_value=[fake, fake])

        exit_code = main(
            [
                "a.pdf",
                "b.pdf",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
            ]
        )

        assert exit_code == 0
        mock_many.assert_called_once()
        captured = capsys.readouterr()
        assert '"name": "Ada"' in captured.out

    def test_batch_defaults_to_abort_on_first_failure(self, mocker):
        fake = MagicMock()
        fake.model_dump.return_value = {"name": "Ada", "age": 36}
        mock_many = _patch_extract_many(mocker, return_value=[fake, fake])

        main(
            [
                "a.pdf",
                "b.pdf",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
            ]
        )

        assert mock_many.call_args.kwargs["return_exceptions"] is False

    def test_continue_on_error_passes_return_exceptions(self, mocker):
        fake = MagicMock()
        fake.model_dump.return_value = {"name": "Ada", "age": 36}
        mock_many = _patch_extract_many(mocker, return_value=[fake, fake])

        main(
            [
                "a.pdf",
                "b.pdf",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--continue-on-error",
            ]
        )

        assert mock_many.call_args.kwargs["return_exceptions"] is True

    def test_continue_on_error_reports_failures_and_exits_7(self, mocker, capsys):
        fake = MagicMock()
        fake.model_dump.return_value = {"name": "Ada", "age": 36}
        _patch_extract_many(mocker, return_value=[fake, ModelError("boom")])

        exit_code = main(
            [
                "a.pdf",
                "b.pdf",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--continue-on-error",
            ]
        )

        assert exit_code == 7
        captured = capsys.readouterr()
        assert '"name": "Ada"' in captured.out  # the successful item is still emitted
        assert '"error_type": "ModelError"' in captured.out
        assert '"input": "b.pdf"' in captured.out
        assert "warning:" not in captured.out
        assert "1 of 2 input(s) failed" in captured.err

    def test_continue_on_error_all_success_exits_0(self, mocker, capsys):
        fake = MagicMock()
        fake.model_dump.return_value = {"name": "Ada", "age": 36}
        _patch_extract_many(mocker, return_value=[fake, fake])

        exit_code = main(
            [
                "a.pdf",
                "b.pdf",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--continue-on-error",
            ]
        )

        assert exit_code == 0
        assert capsys.readouterr().err == ""

    def test_usage_flag_calls_extract_with_usage(self, mocker, capsys):
        from openextract import Usage

        fake = MagicMock()
        fake.model_dump.return_value = {"name": "Ada", "age": 36}
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_usage = _patch_extract_with_usage(mocker, return_value=(fake, usage))

        exit_code = main(
            [
                "input.txt",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--usage",
            ]
        )

        assert exit_code == 0
        mock_usage.assert_called_once()
        captured = capsys.readouterr()
        assert '"usage"' in captured.out
        assert captured.out.count("input_tokens") >= 1

    def test_usage_with_multiple_inputs_returns_1(self, capsys):
        exit_code = main(
            [
                "a.pdf",
                "b.pdf",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--usage",
            ]
        )
        assert exit_code == 1
        assert "exactly one" in capsys.readouterr().err

    def test_stdin_without_media_type_returns_1(self, capsys, mocker):
        mocker.patch("sys.stdin")
        exit_code = main(
            [
                "-",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
            ]
        )
        assert exit_code == 1
        assert "media-type" in capsys.readouterr().err.lower()

    def test_stdin_with_other_paths_returns_1(self, capsys):
        exit_code = main(
            [
                "-",
                "other.pdf",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--media-type",
                "application/pdf",
            ]
        )
        assert exit_code == 1
        assert "stdin" in capsys.readouterr().err.lower()

    def test_stdin_passes_buffer_for_bounded_reading(self, mocker, capsys):
        fake = MagicMock()
        fake.model_dump_json.return_value = "{}"
        mock_extract = _patch_extract(mocker, return_value=fake)
        stdin = mocker.patch("openextract._cli.sys.stdin")

        exit_code = main(
            [
                "-",
                "--schema",
                "tests.test_cli:_FixtureSchema",
                "--model",
                "xai:grok-4.3",
                "--media-type",
                "application/pdf",
            ]
        )

        assert exit_code == 0
        assert mock_extract.call_args.kwargs["input_file"] is stdin.buffer
        assert mock_extract.call_args.kwargs["media_type"] == "application/pdf"
        stdin.buffer.read.assert_not_called()
