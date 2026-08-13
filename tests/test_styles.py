"""Tests for extraction styles."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import BinaryContent

from openextract import (
    AsyncExtractor,
    ExtractionStyle,
    Extractor,
    ModelError,
    ProviderNotInstalledError,
    RetryPolicy,
    extract,
    extract_async,
    extract_many,
    extract_with_usage,
    extract_with_usage_async,
)
from openextract._cli import main
from openextract._styles import (
    decode_text_document,
    document_filename,
    is_binary_media_type,
    is_text_media_type,
    materialize_text_document,
    normalize_style,
    prepared_style_run,
    style_capabilities,
    style_run_inputs,
)
from tests.test_extract import _make_agent_mock, _Person


def _install_harness(mocker, *, filesystem=None, code_mode=None, mount_dir=None):
    harness = SimpleNamespace(
        FileSystem=filesystem or MagicMock(return_value="fs-cap"),
        CodeMode=code_mode or MagicMock(return_value="code-cap"),
    )
    monty = SimpleNamespace(MountDir=mount_dir or MagicMock(return_value="mount"))
    mocker.patch.dict(sys.modules, {"pydantic_ai_harness": harness, "pydantic_monty": monty})
    return harness, monty


class TestStyleHelpers:
    def test_normalize_accepts_enum_and_string(self):
        assert normalize_style("search") is ExtractionStyle.SEARCH
        assert normalize_style(ExtractionStyle.CODE) is ExtractionStyle.CODE

    def test_normalize_rejects_unknown(self):
        with pytest.raises(ValueError, match="style must be one of"):
            normalize_style("rag")

    @pytest.mark.parametrize(
        ("media_type", "filename"),
        [
            ("text/plain", "document.txt"),
            ("application/json; charset=utf-8", "document.json"),
            ("text/markdown", "document.md"),
            ("application/yaml", "document.yaml"),
            ("text/unknown", "document.txt"),
        ],
    )
    def test_document_filename(self, media_type, filename):
        assert document_filename(media_type) == filename

    def test_text_and_binary_media_types(self):
        assert is_text_media_type("text/plain")
        assert is_text_media_type("application/json")
        assert is_text_media_type("application/vnd.api+json")
        assert is_binary_media_type("image/png")
        assert is_binary_media_type("application/pdf")
        assert not is_binary_media_type("text/plain")

    def test_office_documents_are_binary(self):
        assert is_binary_media_type(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert is_binary_media_type(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert is_binary_media_type("application/vnd.oasis.opendocument.text")
        assert is_binary_media_type("application/msword")
        assert is_binary_media_type("application/vnd.ms-excel")
        assert is_binary_media_type("application/vnd.ms-powerpoint")

    def test_decode_rejects_docx_with_direct_guidance(self):
        docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        with pytest.raises(ValueError, match="Use style='direct'"):
            decode_text_document(b"PK\x03\x04", docx, style=ExtractionStyle.SEARCH)

    def test_decode_rejects_pdf_and_nul_and_invalid_utf8(self):
        with pytest.raises(ValueError, match="style 'search' requires a text document"):
            decode_text_document(b"%PDF", "application/pdf", style=ExtractionStyle.SEARCH)
        with pytest.raises(ValueError, match="NUL bytes"):
            decode_text_document(b"a\x00b", "text/plain", style=ExtractionStyle.CODE)
        with pytest.raises(ValueError, match="valid UTF-8"):
            decode_text_document(b"\xff", "text/plain", style=ExtractionStyle.SEARCH)

    def test_decode_accepts_octet_stream_utf8(self):
        assert (
            decode_text_document(b"hello", "application/octet-stream", style=ExtractionStyle.SEARCH)
            == "hello"
        )

    def test_materialize_writes_workspace_file(self, tmp_path):
        name = materialize_text_document(
            tmp_path, b'{"a": 1}', "application/json", style=ExtractionStyle.CODE
        )
        assert name == "document.json"
        assert (tmp_path / name).read_text(encoding="utf-8") == '{"a": 1}'

    def test_direct_capabilities_are_empty(self, tmp_path):
        assert style_capabilities(ExtractionStyle.DIRECT, tmp_path) == []

    def test_style_run_inputs_mention_tools_or_code(self):
        search = style_run_inputs(ExtractionStyle.SEARCH, "document.txt")
        code = style_run_inputs(ExtractionStyle.CODE, "document.txt")
        assert "search_files" in search[0]
        assert "/work/document.txt" in code[0]


class TestPreparedStyleRun:
    def test_direct_yields_no_workspace_inputs(self):
        with prepared_style_run(ExtractionStyle.DIRECT, b"abc", "text/plain") as (
            caps,
            inputs,
        ):
            assert caps == []
            assert inputs is None

    def test_search_materializes_and_cleans_up(self, mocker):
        harness, _ = _install_harness(mocker)
        with prepared_style_run(ExtractionStyle.SEARCH, b"body", "text/plain") as (
            caps,
            inputs,
        ):
            root = harness.FileSystem.call_args.kwargs["root_dir"]
            assert (root / "document.txt").read_text(encoding="utf-8") == "body"
            assert caps == [harness.FileSystem.return_value]
            assert "search_files" in inputs[0]
        assert not root.exists()


class TestExtractStyles:
    def test_invalid_style_is_rejected_before_agent_build(self, mocker):
        agent = mocker.patch("openextract._extract.Agent")
        with pytest.raises(ValueError, match="style must be one of"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file=b"x",
                media_type="text/plain",
                style="nope",
            )
        agent.assert_not_called()

    def test_search_requires_harness_extra(self, mocker):
        mocker.patch.dict(sys.modules, {"pydantic_ai_harness": None})
        mocker.patch("openextract._extract.Agent")
        with pytest.raises(ProviderNotInstalledError, match="pydantic-ai-harness"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file=b"Ada is 36",
                media_type="text/plain",
                style="search",
            )

    def test_code_requires_code_extra(self, mocker):
        mocker.patch.dict(sys.modules, {"pydantic_ai_harness": None, "pydantic_monty": None})
        mocker.patch("openextract._extract.Agent")
        with pytest.raises(ProviderNotInstalledError, match="codemode"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file=b"Ada is 36",
                media_type="text/plain",
                style="code",
            )

    def test_search_passes_filesystem_capability_not_binary(self, mocker):
        harness, _ = _install_harness(mocker)
        expected = _Person(name="Ada", age=36)
        agent_cls, agent = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=b"Ada is 36",
            media_type="text/plain",
            style="search",
        )

        assert result is expected
        assert agent_cls.call_args.kwargs["capabilities"] == [harness.FileSystem.return_value]
        prompt = agent.run_sync.call_args.args[0]
        assert all(not isinstance(part, BinaryContent) for part in prompt)
        assert "search_files" in prompt[0]

    def test_code_mounts_workspace_read_only(self, mocker):
        harness, monty = _install_harness(mocker)
        expected = _Person(name="Ada", age=36)
        agent_cls, agent = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=b"Ada is 36",
            media_type="text/plain",
            style=ExtractionStyle.CODE,
        )

        assert result is expected
        monty.MountDir.assert_called_once()
        assert monty.MountDir.call_args.kwargs["virtual_path"] == "/work"
        assert monty.MountDir.call_args.kwargs["mode"] == "read-only"
        harness.CodeMode.assert_called_once_with(mount=monty.MountDir.return_value)
        assert agent_cls.call_args.kwargs["capabilities"] == [harness.CodeMode.return_value]
        assert "/work/document.txt" in agent.run_sync.call_args.args[0][0]

    def test_search_rejects_images(self, mocker):
        mocker.patch("openextract._extract.Agent")
        with pytest.raises(ValueError, match="style 'search' requires a text document"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file=b"\x89PNG",
                media_type="image/png",
                style="search",
            )

    def test_usage_and_async_search(self, mocker):
        import asyncio
        from unittest.mock import AsyncMock

        _install_harness(mocker)
        expected = _Person(name="Ada", age=36)
        usage = SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3)
        _, agent = _make_agent_mock(mocker, output=expected, usage=usage)
        run_result = agent.run_sync.return_value
        agent.run = AsyncMock(return_value=run_result)

        output, tokens = extract_with_usage(
            schema=_Person,
            model="openai:gpt-5",
            input_file=b"Ada",
            media_type="text/plain",
            style="search",
        )
        assert output is expected
        assert tokens.total_tokens == 3

        async_output, async_tokens = asyncio.run(
            extract_with_usage_async(
                schema=_Person,
                model="openai:gpt-5",
                input_file=b"Ada",
                media_type="text/plain",
                style="search",
            )
        )
        assert async_output is expected
        assert async_tokens.total_tokens == 3

        assert (
            asyncio.run(
                extract_async(
                    schema=_Person,
                    model="openai:gpt-5",
                    input_file=b"Ada",
                    media_type="text/plain",
                    style="search",
                )
            )
            is expected
        )

    def test_search_workspace_survives_retries(self, mocker):
        harness, _ = _install_harness(mocker)
        expected = _Person(name="Ada", age=36)
        attempts: list[str] = []

        def run_sync(inputs):
            root = harness.FileSystem.call_args.kwargs["root_dir"]
            attempts.append((root / "document.txt").read_text(encoding="utf-8"))
            if len(attempts) == 1:
                raise ModelError("transient", status_code=503)
            return SimpleNamespace(output=expected)

        _make_agent_mock(mocker, run_sync_side_effect=run_sync)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=b"Ada is 36",
            media_type="text/plain",
            style="search",
            max_retries=1,
            retry_backoff=0,
        )

        assert result is expected
        # The same materialized document was readable on both attempts.
        assert attempts == ["Ada is 36", "Ada is 36"]
        root = harness.FileSystem.call_args.kwargs["root_dir"]
        assert not root.exists()

    def test_batch_search_builds_one_agent_per_item(self, mocker):
        _install_harness(mocker)
        people = [_Person(name="Ada", age=36), _Person(name="Grace", age=85)]
        agent = MagicMock()
        agent.run = MagicMock()

        async def run(inputs):
            return SimpleNamespace(output=people.pop(0))

        agent.run.side_effect = run
        build = mocker.patch("openextract._extract._build_agent", return_value=agent)

        results = extract_many(
            schema=_Person,
            model="openai:gpt-5",
            input_files=[b"a", b"b"],
            media_type="text/plain",
            style="search",
        )

        assert [item.name for item in results] == ["Ada", "Grace"]
        assert build.call_count == 2
        assert all(call.kwargs["extra_capabilities"] for call in build.call_args_list)


class TestSessionStyles:
    def test_injected_agent_rejects_non_direct_style(self):
        with pytest.raises(ValueError, match="injected agent"):
            Extractor(_Person, agent=MagicMock(), style="search")

    def test_search_session_builds_agent_once_on_enter(self, mocker):
        from tests.test_sessions import FakeAgent

        harness, _ = _install_harness(mocker)
        agent = FakeAgent([{"name": "Ada", "age": 36}])
        build = mocker.patch("openextract._extract._build_agent", return_value=agent)
        extractor = Extractor(_Person, "openai:gpt-5", style="search")
        build.assert_not_called()
        with extractor:
            extractor.extract(b"Ada", media_type="text/plain")
        build.assert_called_once()
        assert build.call_args.kwargs["extra_capabilities"] == [harness.FileSystem.return_value]
        assert agent.enter_count == 1
        assert agent.exit_count == 1

    def test_sync_search_session_extracts(self, mocker):
        from tests.test_sessions import FakeAgent

        harness, _ = _install_harness(mocker)
        agent = FakeAgent([{"name": "Ada", "age": 36}, {"name": "Grace", "age": 85}])
        mocker.patch("openextract._extract._build_agent", return_value=agent)
        with Extractor(_Person, "openai:gpt-5", style="search") as extractor:
            first = extractor.extract(b"Ada", media_type="text/plain")
            second, usage = extractor.extract_with_usage(b"Grace", media_type="text/plain")
            workspace = harness.FileSystem.call_args.kwargs["root_dir"]
            # Per-call documents are removed; the workspace lives until close.
            assert workspace.exists()
            assert list(workspace.iterdir()) == []
        assert first == _Person(name="Ada", age=36)
        assert second == _Person(name="Grace", age=85)
        assert usage.total_tokens == 3
        assert not workspace.exists()

    def test_sync_search_session_workspace_survives_retries(self, mocker):
        from tests.test_sessions import FakeAgent

        harness, _ = _install_harness(mocker)
        documents: list[str] = []

        class WorkspaceProbeAgent(FakeAgent):
            async def run(self, inputs):
                root = harness.FileSystem.call_args.kwargs["root_dir"]
                filename = inputs[0].split("'")[1]
                documents.append((root / filename).read_text(encoding="utf-8"))
                return await super().run(inputs)

        agent = WorkspaceProbeAgent(
            [ModelError("transient", status_code=503), {"name": "Ada", "age": 36}]
        )
        mocker.patch("openextract._extract._build_agent", return_value=agent)
        policy = RetryPolicy(max_retries=1, backoff=0)
        with Extractor(_Person, "openai:gpt-5", style="search", retry_policy=policy) as session:
            result = session.extract(b"Ada is 36", media_type="text/plain")
        assert result == _Person(name="Ada", age=36)
        assert agent.run_count == 2
        # The same materialized document was readable on both attempts.
        assert documents == ["Ada is 36", "Ada is 36"]

    def test_sync_search_session_enter_failure_cleans_up(self, mocker):
        mocker.patch.dict(sys.modules, {"pydantic_ai_harness": None})
        client = MagicMock()
        mocker.patch("openextract._extract.httpx.Client", return_value=client)
        workspace_spy = mocker.spy(tempfile, "TemporaryDirectory")
        extractor = Extractor(_Person, "openai:gpt-5", style="search")
        with pytest.raises(ProviderNotInstalledError, match="pydantic-ai-harness"):
            extractor.__enter__()
        client.close.assert_called_once_with()
        assert not Path(workspace_spy.spy_return.name).exists()

    async def test_async_code_session_enter_failure_closes_client(self, mocker):
        mocker.patch.dict(sys.modules, {"pydantic_ai_harness": None, "pydantic_monty": None})
        client = MagicMock()
        client.aclose = AsyncMock()
        mocker.patch("openextract._extract.httpx.AsyncClient", return_value=client)
        extractor = AsyncExtractor(_Person, "openai:gpt-5", style="code")
        with pytest.raises(ProviderNotInstalledError, match="codemode"):
            await extractor.__aenter__()
        client.aclose.assert_awaited_once_with()

    async def test_async_code_session_extracts(self, mocker):
        from tests.test_sessions import FakeAgent

        _install_harness(mocker)
        agent = FakeAgent([{"name": "Ada", "age": 36}, {"name": "Grace", "age": 85}])
        mocker.patch("openextract._extract._build_agent", return_value=agent)
        async with AsyncExtractor(_Person, "openai:gpt-5", style="code") as extractor:
            first = await extractor.extract(b"Ada", media_type="text/plain")
            second, usage = await extractor.extract_with_usage(b"Grace", media_type="text/plain")
        assert first == _Person(name="Ada", age=36)
        assert second == _Person(name="Grace", age=85)
        assert usage.total_tokens == 3


class TestCliStyle:
    def test_style_is_forwarded(self, mocker):
        fake = _Person(name="Ada", age=36)
        mock_extract = mocker.patch("openextract._cli.extract", return_value=fake)

        assert (
            main(
                [
                    "input.txt",
                    "--schema",
                    "tests.test_cli:_FixtureSchema",
                    "--model",
                    "openai:gpt-5",
                    "--style",
                    "search",
                ]
            )
            == 0
        )
        assert mock_extract.call_args.kwargs["style"] == "search"
