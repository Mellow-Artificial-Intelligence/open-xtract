"""Tests for openextract._extract."""

import asyncio
import io
import ipaddress
import socket
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai import BinaryContent

from openextract import (
    ExtractionError,
    ModelError,
    SchemaValidationError,
    UrlFetchError,
    Usage,
    extract,
    extract_async,
    extract_many,
    extract_many_async,
    extract_with_usage,
)
from openextract._extract import (
    _fetch_url,
    _get_media,
    _get_media_type,
    _is_public_ip,
    _is_safe_host,
)


def _build_response(
    *,
    content: bytes = b"",
    content_type: str = "application/octet-stream",
    is_redirect: bool = False,
    status_code: int = 200,
    location: str | None = None,
) -> MagicMock:
    """Build a MagicMock that behaves like an httpx.Response."""
    response = MagicMock()
    response.content = content
    headers: dict[str, str] = {}
    if content_type:
        headers["content-type"] = content_type
    if location is not None:
        headers["location"] = location
    response.headers = headers
    response.is_redirect = is_redirect
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_star_import_exposes_only_existing_names():
    """`from openextract import *` must not reference names that aren't defined."""
    namespace: dict = {}
    exec("from openextract import *", namespace)
    exported = {name for name in namespace if not name.startswith("_")}
    assert exported == {
        "extract",
        "extract_async",
        "extract_many",
        "extract_many_async",
        "extract_with_usage",
        "Usage",
        "ExtractionError",
        "ModelError",
        "SchemaValidationError",
        "UrlFetchError",
    }


# ---------------------------------------------------------------------------
# _get_media_type
# ---------------------------------------------------------------------------


class TestGetMediaType:
    def test_returns_png_mime_type(self):
        assert _get_media_type("image.png") == "image/png"

    def test_returns_jpeg_mime_type(self):
        assert _get_media_type("photo.jpg") == "image/jpeg"

    def test_returns_text_mime_type(self):
        assert _get_media_type("notes.txt") == "text/plain"

    def test_returns_octet_stream_for_unknown_extension(self):
        assert _get_media_type("mystery.xyz123") == "application/octet-stream"

    def test_returns_octet_stream_for_no_extension(self):
        assert _get_media_type("README") == "application/octet-stream"

    def test_handles_full_path(self):
        assert _get_media_type("/tmp/nested/dir/file.png") == "image/png"


# ---------------------------------------------------------------------------
# _get_media
# ---------------------------------------------------------------------------


class TestGetMedia:
    def test_reads_local_file_with_known_extension(self, tmp_path):
        local = tmp_path / "hello.txt"
        local.write_bytes(b"hello world")

        media_bytes, media_type = _get_media(str(local))

        assert media_bytes == b"hello world"
        assert media_type == "text/plain"

    def test_reads_local_file_with_unknown_extension(self, tmp_path):
        odd_file = tmp_path / "data.weirdext"
        odd_file.write_bytes(b"opaque-bytes")

        media_bytes, media_type = _get_media(str(odd_file))

        assert media_bytes == b"opaque-bytes"
        assert media_type == "application/octet-stream"

    def test_reads_project_pdf_fixture(self):
        media_bytes, media_type = _get_media("tests/test.pdf")

        assert media_bytes.startswith(b"%PDF")
        assert media_type == "application/pdf"

    def test_missing_local_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            _get_media(str(missing))

    def test_fetches_https_url(self, mocker):
        fake_response = _build_response(content=b"<html>remote</html>")
        mock_get = mocker.patch("openextract._extract.httpx.get", return_value=fake_response)

        media_bytes, media_type = _get_media("https://example.com/page.html")

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://example.com/page.html"
        # follow_redirects is disabled at the httpx layer; redirects are followed
        # manually in _fetch_url so the SSRF host check runs at every hop.
        assert kwargs["follow_redirects"] is False
        assert kwargs["timeout"] == 30.0
        assert media_bytes == b"<html>remote</html>"
        assert media_type == "text/html"

    def test_fetches_http_url(self, mocker):
        """http:// URLs are fetched, not treated as local paths."""
        fake_response = _build_response(content=b"plain")
        mocker.patch("openextract._extract.httpx.get", return_value=fake_response)

        media_bytes, media_type = _get_media("http://example.com/page.html")

        assert media_bytes == b"plain"
        assert media_type == "text/html"

    def test_url_without_useful_extension_falls_back_to_response_header(self, mocker):
        fake_response = _build_response(
            content=b"raw-bytes",
            content_type="application/pdf; charset=binary",
        )
        mocker.patch("openextract._extract.httpx.get", return_value=fake_response)

        media_bytes, media_type = _get_media("https://example.com/download?id=42")

        assert media_bytes == b"raw-bytes"
        assert media_type == "application/pdf"

    def test_url_with_no_extension_and_no_header_stays_octet_stream(self, mocker):
        fake_response = _build_response(content=b"raw-bytes", content_type="")
        mocker.patch("openextract._extract.httpx.get", return_value=fake_response)

        _, media_type = _get_media("https://example.com/blob")

        assert media_type == "application/octet-stream"

    def test_known_url_extension_ignores_response_header(self, mocker):
        """URL extension wins when it's specific; protects against misconfigured servers."""
        fake_response = _build_response(content=b"%PDF", content_type="text/html")
        mocker.patch("openextract._extract.httpx.get", return_value=fake_response)

        _, media_type = _get_media("https://example.com/doc.pdf")

        assert media_type == "application/pdf"

    def test_http_error_status_raises(self, mocker):
        fake_response = _build_response(content=b"<html>404</html>")
        fake_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=fake_response
        )
        mocker.patch("openextract._extract.httpx.get", return_value=fake_response)

        with pytest.raises(httpx.HTTPStatusError):
            _get_media("https://example.com/missing.pdf")


# ---------------------------------------------------------------------------
# SSRF host validation
# ---------------------------------------------------------------------------


def _addrinfo(ip: str):
    """Build a getaddrinfo-shaped tuple for ``ip``."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 0, "", (ip, 0))]


class TestIsPublicIp:
    def test_public_ipv4_is_public(self):
        assert _is_public_ip(ipaddress.ip_address("1.1.1.1")) is True

    def test_private_ipv4_is_not_public(self):
        assert _is_public_ip(ipaddress.ip_address("10.0.0.1")) is False

    def test_loopback_ipv4_is_not_public(self):
        assert _is_public_ip(ipaddress.ip_address("127.0.0.1")) is False

    def test_aws_metadata_link_local_is_not_public(self):
        assert _is_public_ip(ipaddress.ip_address("169.254.169.254")) is False

    def test_ipv6_loopback_is_not_public(self):
        assert _is_public_ip(ipaddress.ip_address("::1")) is False

    def test_ipv4_mapped_ipv6_loopback_is_not_public(self):
        # ::ffff:127.0.0.1 wraps an IPv4 loopback; must be unwrapped and rejected.
        assert _is_public_ip(ipaddress.ip_address("::ffff:127.0.0.1")) is False


class TestIsSafeHost:
    def test_opt_out_env_var_bypasses_validation(self, monkeypatch):
        monkeypatch.setenv("OPENEXTRACT_ALLOW_PRIVATE_URLS", "1")
        # Even a clearly private address is permitted when the opt-out is set.
        assert _is_safe_host("127.0.0.1") is True

    def test_opt_out_env_var_unset_does_not_bypass(self, monkeypatch):
        monkeypatch.delenv("OPENEXTRACT_ALLOW_PRIVATE_URLS", raising=False)
        assert _is_safe_host("127.0.0.1") is False

    def test_empty_host_is_unsafe(self):
        assert _is_safe_host("") is False
        assert _is_safe_host(None) is False

    def test_private_ip_literal_is_unsafe(self):
        assert _is_safe_host("192.168.1.1") is False
        assert _is_safe_host("169.254.169.254") is False

    def test_public_ip_literal_is_safe(self):
        assert _is_safe_host("1.1.1.1") is True

    def test_ipv6_literal_in_brackets_is_handled(self):
        # urlparse strips brackets, but defend in depth: _is_safe_host must too.
        assert _is_safe_host("[::1]") is False

    def test_hostname_resolving_to_private_ip_is_unsafe(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("10.0.0.5"))
        assert _is_safe_host("internal.example.com") is False

    def test_hostname_with_dns_failure_is_unsafe(self, monkeypatch):
        def boom(*args, **kwargs):
            raise socket.gaierror("nope")

        monkeypatch.setattr(socket, "getaddrinfo", boom)
        assert _is_safe_host("nonexistent.invalid") is False

    def test_hostname_with_empty_resolution_is_unsafe(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
        assert _is_safe_host("ghost.example.com") is False

    def test_hostname_with_unparseable_resolved_address_is_unsafe(self, monkeypatch):
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("not-an-ip", 0))],
        )
        assert _is_safe_host("weird.example.com") is False


class TestFetchUrl:
    def test_refuses_private_host(self):
        with pytest.raises(UrlFetchError, match="non-public host"):
            _fetch_url("http://127.0.0.1/secret")

    def test_refuses_aws_metadata_url(self):
        with pytest.raises(UrlFetchError, match="non-public host"):
            _fetch_url("http://169.254.169.254/latest/meta-data/")

    def test_follows_safe_redirect(self, mocker):
        redirect = _build_response(
            content=b"", status_code=302, is_redirect=True, location="https://2.2.2.2/final.html"
        )
        final = _build_response(content=b"ok", content_type="text/html")
        mock_get = mocker.patch("openextract._extract.httpx.get", side_effect=[redirect, final])

        response = _fetch_url("https://1.1.1.1/start")

        assert response is final
        assert mock_get.call_count == 2
        # Both hops must be issued with follow_redirects disabled so our host
        # check runs each time.
        for call in mock_get.call_args_list:
            assert call.kwargs["follow_redirects"] is False

    def test_blocks_redirect_to_private_host(self, mocker):
        redirect = _build_response(
            content=b"",
            status_code=302,
            is_redirect=True,
            location="http://169.254.169.254/latest/meta-data/",
        )
        mocker.patch("openextract._extract.httpx.get", return_value=redirect)

        with pytest.raises(UrlFetchError, match="non-public host"):
            _fetch_url("https://1.1.1.1/start")

    def test_redirect_without_location_raises_url_fetch_error(self, mocker):
        no_location = _build_response(content=b"", status_code=302, is_redirect=True, location=None)
        mocker.patch("openextract._extract.httpx.get", return_value=no_location)

        with pytest.raises(UrlFetchError, match="missing Location"):
            _fetch_url("https://1.1.1.1/start")

    def test_too_many_redirects_raises(self, mocker):
        redirect = _build_response(
            content=b"", status_code=302, is_redirect=True, location="https://1.1.1.1/loop"
        )
        mocker.patch("openextract._extract.httpx.get", return_value=redirect)

        with pytest.raises(UrlFetchError, match="Too many redirects"):
            _fetch_url("https://1.1.1.1/loop")


class TestSsrfIntegration:
    def test_extract_with_private_url_raises_url_fetch_error(self, mocker):
        # Agent should never be constructed when the URL is rejected.
        agent_cls = mocker.patch("openextract._extract.Agent")
        with pytest.raises(UrlFetchError):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="http://169.254.169.254/latest/meta-data/",
            )
        agent_cls.assert_not_called()


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


class _Person(BaseModel):
    name: str
    age: int


def _make_agent_mock(mocker, output=None, run_sync_side_effect=None, usage=None):
    """Patch openextract._extract.Agent and return (AgentClass, instance, run_sync)."""
    agent_instance = MagicMock()
    if run_sync_side_effect is not None:
        agent_instance.run_sync.side_effect = run_sync_side_effect
    else:
        run_result = MagicMock()
        run_result.output = output
        if usage is not None:
            run_result.usage.return_value = usage
        agent_instance.run_sync.return_value = run_result
    agent_cls = mocker.patch("openextract._extract.Agent", return_value=agent_instance)
    return agent_cls, agent_instance


class TestExtract:
    def test_returns_schema_instance_from_local_file(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Ada", age=36)
        agent_cls, agent_instance = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            instructions="pull the person",
        )

        assert result is expected
        agent_cls.assert_called_once()
        # Non-ollama models pass the schema directly as output_type.
        kwargs = agent_cls.call_args.kwargs
        assert kwargs["instructions"] == "pull the person"
        assert kwargs["output_type"] is _Person
        agent_instance.run_sync.assert_called_once()

    def test_ollama_model_wraps_output_in_native_output(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Linus", age=54)
        agent_cls, _ = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="ollama:llama3",
            input_file=str(local),
        )

        assert result is expected
        output_type = agent_cls.call_args.kwargs["output_type"]
        # NativeOutput wraps the schema; assert it's not the bare schema class.
        assert output_type is not _Person
        assert type(output_type).__name__ == "NativeOutput"

    def test_http_status_error_is_wrapped(self, mocker):
        response = MagicMock()
        response.status_code = 503
        err = httpx.HTTPStatusError("boom", request=MagicMock(), response=response)
        mocker.patch("openextract._extract._get_media", side_effect=err)

        with pytest.raises(UrlFetchError, match="503"):
            extract(schema=_Person, model="openai:gpt-5", input_file="https://x/y")

    def test_request_error_is_wrapped(self, mocker):
        err = httpx.ConnectError("dns failure")
        mocker.patch("openextract._extract._get_media", side_effect=err)

        with pytest.raises(UrlFetchError, match="dns failure"):
            extract(schema=_Person, model="openai:gpt-5", input_file="https://x/y")

    def test_http_404_from_url_becomes_url_fetch_error(self, mocker):
        """End-to-end: a 404 response from the URL fetch surfaces as UrlFetchError, not garbage."""
        fake_response = _build_response(content=b"<html>not found</html>")
        fake_response.status_code = 404
        fake_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=fake_response
        )
        mocker.patch("openextract._extract.httpx.get", return_value=fake_response)

        with pytest.raises(UrlFetchError, match="404"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="https://example.com/missing.pdf",
            )

    def test_validation_error_is_wrapped(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        try:
            _Person(name="x", age="not-an-int")  # type: ignore[arg-type]
        except ValidationError as exc:
            validation_error = exc
        _make_agent_mock(mocker, run_sync_side_effect=validation_error)

        with pytest.raises(SchemaValidationError, match="Model output did not match schema"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_pydantic_ai_model_api_error_is_wrapped_as_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        from pydantic_ai.exceptions import ModelHTTPError

        provider_error = ModelHTTPError(status_code=503, model_name="gpt-5", body="upstream down")
        _make_agent_mock(mocker, run_sync_side_effect=provider_error)

        with pytest.raises(ModelError, match="Model API error"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_openai_api_error_is_wrapped_as_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        from openai import APIError as OpenAIAPIError

        class _FakeOpenAIError(OpenAIAPIError):
            def __init__(self, message: str):
                # Bypass OpenAIAPIError.__init__ to avoid constructing request/body objects.
                Exception.__init__(self, message)

        _make_agent_mock(mocker, run_sync_side_effect=_FakeOpenAIError("rate limited"))

        with pytest.raises(ModelError, match="Model API error"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_anthropic_api_error_is_wrapped_as_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        from anthropic import APIError as AnthropicAPIError

        class _FakeAnthropicError(AnthropicAPIError):
            def __init__(self, message: str):
                # Bypass AnthropicAPIError.__init__ to avoid constructing request/body objects.
                Exception.__init__(self, message)

        _make_agent_mock(mocker, run_sync_side_effect=_FakeAnthropicError("rate limited"))

        with pytest.raises(ModelError, match="Model API error"):
            extract(schema=_Person, model="anthropic:claude-sonnet-4", input_file=str(local))

    def test_bedrock_client_error_is_wrapped_as_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        from botocore.exceptions import ClientError as BedrockClientError

        provider_error = BedrockClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "rate limited"}},
            "InvokeModel",
        )
        _make_agent_mock(mocker, run_sync_side_effect=provider_error)

        with pytest.raises(ModelError, match="Model API error"):
            extract(
                schema=_Person,
                model="bedrock:anthropic.claude-sonnet-4-20250514-v1:0",
                input_file=str(local),
            )

    def test_cohere_api_error_is_wrapped_as_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        from cohere.core.api_error import ApiError as CohereApiError

        class _FakeCohereError(CohereApiError):
            def __init__(self, message: str):
                # Bypass CohereApiError.__init__; populate the attributes its __str__ needs.
                Exception.__init__(self, message)
                self.headers = None
                self.status_code = 401
                self.body = message

        _make_agent_mock(mocker, run_sync_side_effect=_FakeCohereError("unauthorized"))

        with pytest.raises(ModelError, match="Model API error"):
            extract(schema=_Person, model="cohere:command-r-plus", input_file=str(local))

    def test_huggingface_http_error_is_wrapped_as_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        from huggingface_hub.errors import HfHubHTTPError

        class _FakeHfHubHTTPError(HfHubHTTPError):
            def __init__(self, message: str):
                # Bypass HfHubHTTPError.__init__ to avoid constructing a Response object.
                Exception.__init__(self, message)

        _make_agent_mock(mocker, run_sync_side_effect=_FakeHfHubHTTPError("hf upstream down"))

        with pytest.raises(ModelError, match="Model API error"):
            extract(
                schema=_Person,
                model="huggingface:meta-llama/Llama-3.3-70B-Instruct",
                input_file=str(local),
            )

    def test_groq_api_error_is_wrapped_as_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        from groq import APIError as GroqAPIError

        class _FakeGroqError(GroqAPIError):
            def __init__(self, message: str):
                # Bypass GroqAPIError.__init__ to avoid constructing request/body objects.
                Exception.__init__(self, message)

        _make_agent_mock(mocker, run_sync_side_effect=_FakeGroqError("rate limited"))

        with pytest.raises(ModelError, match="Model API error"):
            extract(schema=_Person, model="groq:llama-3.3-70b-versatile", input_file=str(local))

    def test_mistral_sdk_error_is_wrapped_as_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        from mistralai.client.errors.sdkerror import SDKError as MistralSDKError

        class _FakeMistralError(MistralSDKError):
            def __init__(self, message: str):
                # Bypass SDKError.__init__ to avoid constructing httpx response objects.
                Exception.__init__(self, message)
                object.__setattr__(self, "message", message)

        _make_agent_mock(mocker, run_sync_side_effect=_FakeMistralError("rate limited"))

        with pytest.raises(ModelError, match="Model API error"):
            extract(schema=_Person, model="mistral:mistral-large-latest", input_file=str(local))

    def test_message_mentioning_model_is_wrapped_as_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(
            mocker,
            run_sync_side_effect=RuntimeError("unknown model identifier"),
        )

        with pytest.raises(ExtractionError, match="Extraction failed: unknown model identifier"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_generic_exception_is_wrapped_as_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(mocker, run_sync_side_effect=RuntimeError("kaboom"))

        with pytest.raises(ExtractionError, match="Extraction failed: kaboom"):
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_message_mentioning_model_is_not_subclass_of_model_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(
            mocker,
            run_sync_side_effect=RuntimeError("the model said no"),
        )

        with pytest.raises(ExtractionError) as exc_info:
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))
        # The new classifier should NOT promote this to ModelError just because
        # the message mentions "model".
        assert not isinstance(exc_info.value, ModelError)

    def test_passes_through_existing_extraction_error(self, tmp_path, mocker):
        """If the wrapped code already raises an ExtractionError, it is not re-wrapped."""
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        original = ModelError("already mapped")
        _make_agent_mock(mocker, run_sync_side_effect=original)

        with pytest.raises(ModelError) as exc_info:
            extract(schema=_Person, model="openai:gpt-5", input_file=str(local))
        assert exc_info.value is original

    def test_extract_returns_only_model_instance(self, tmp_path, mocker):
        """Regression: extract() still returns just the schema instance, not a tuple."""
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Grace", age=42)
        usage_obj = MagicMock(input_tokens=1, output_tokens=2, total_tokens=3)
        _make_agent_mock(mocker, output=expected, usage=usage_obj)

        result = extract(schema=_Person, model="openai:gpt-5", input_file=str(local))

        assert result is expected
        assert not isinstance(result, tuple)


# ---------------------------------------------------------------------------
# extract: input_file polymorphism
# ---------------------------------------------------------------------------


def _binary_content_arg(agent_instance) -> BinaryContent:
    """Pull the BinaryContent that was passed to agent.run_sync."""
    (call_args,) = agent_instance.run_sync.call_args.args
    binary = next(part for part in call_args if isinstance(part, BinaryContent))
    return binary


class TestExtractInputs:
    def test_bytes_input_with_media_type(self, mocker):
        expected = _Person(name="Grace", age=85)
        _, agent_instance = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=b"raw-payload",
            media_type="application/pdf",
        )

        assert result is expected
        binary = _binary_content_arg(agent_instance)
        assert binary.data == b"raw-payload"
        assert binary.media_type == "application/pdf"

    def test_filelike_input_with_media_type(self, mocker):
        expected = _Person(name="Ada", age=36)
        _, agent_instance = _make_agent_mock(mocker, output=expected)
        buffer = io.BytesIO(b"buffered-bytes")

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=buffer,
            media_type="image/png",
        )

        assert result is expected
        binary = _binary_content_arg(agent_instance)
        assert binary.data == b"buffered-bytes"
        assert binary.media_type == "image/png"
        # Caller owns the handle; we must not close it.
        assert not buffer.closed

    def test_bytes_input_without_media_type_raises(self, mocker):
        _make_agent_mock(mocker, output=_Person(name="x", age=1))

        with pytest.raises(TypeError, match="media_type is required"):
            extract(schema=_Person, model="openai:gpt-5", input_file=b"abc")

    def test_filelike_input_without_media_type_raises(self, mocker):
        _make_agent_mock(mocker, output=_Person(name="x", age=1))

        with pytest.raises(TypeError, match="media_type is required"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file=io.BytesIO(b"abc"),
            )

    def test_str_input_still_works(self, tmp_path, mocker):
        local = tmp_path / "doc.txt"
        local.write_bytes(b"file-bytes")
        expected = _Person(name="Linus", age=54)
        _, agent_instance = _make_agent_mock(mocker, output=expected)

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
        )

        assert result is expected
        binary = _binary_content_arg(agent_instance)
        assert binary.data == b"file-bytes"
        assert binary.media_type == "text/plain"

    def test_str_input_media_type_override(self, tmp_path, mocker):
        local = tmp_path / "doc.txt"
        local.write_bytes(b"file-bytes")
        _, agent_instance = _make_agent_mock(mocker, output=_Person(name="x", age=1))

        extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            media_type="text/markdown",
        )

        binary = _binary_content_arg(agent_instance)
        assert binary.media_type == "text/markdown"

    def test_unsupported_input_type_raises_type_error(self, mocker):
        _make_agent_mock(mocker, output=_Person(name="x", age=1))

        with pytest.raises(TypeError, match="input_file must be"):
            extract(schema=_Person, model="openai:gpt-5", input_file=12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_with_usage
# ---------------------------------------------------------------------------


class TestExtractWithUsage:
    def test_returns_output_and_usage_tuple(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Ada", age=36)
        usage_obj = MagicMock(input_tokens=10, output_tokens=20, total_tokens=30)
        _make_agent_mock(mocker, output=expected, usage=usage_obj)

        output, usage = extract_with_usage(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            instructions="pull the person",
        )

        assert output is expected
        assert usage == Usage(input_tokens=10, output_tokens=20, total_tokens=30)

    def test_missing_usage_fields_default_to_zero(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Linus", age=54)

        class _BareUsage:
            pass

        _make_agent_mock(mocker, output=expected, usage=_BareUsage())

        _, usage = extract_with_usage(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
        )

        assert usage == Usage(input_tokens=0, output_tokens=0, total_tokens=0)

    def test_none_usage_fields_default_to_zero(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        expected = _Person(name="Hedy", age=29)
        usage_obj = MagicMock(input_tokens=None, output_tokens=None, total_tokens=None)
        _make_agent_mock(mocker, output=expected, usage=usage_obj)

        _, usage = extract_with_usage(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
        )

        assert usage == Usage(input_tokens=0, output_tokens=0, total_tokens=0)

    def test_propagates_runtime_error_as_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hello")
        _make_agent_mock(mocker, run_sync_side_effect=RuntimeError("kaboom"))

        with pytest.raises(ExtractionError, match="Extraction failed: kaboom"):
            extract_with_usage(schema=_Person, model="openai:gpt-5", input_file=str(local))

    def test_usage_is_frozen_dataclass(self):
        usage = Usage(input_tokens=1, output_tokens=2, total_tokens=3)
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError varies by py version
            usage.input_tokens = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# extract retry behavior
# ---------------------------------------------------------------------------


class TestExtractRetry:
    def test_no_retry_by_default_raises_immediately(self, mocker):
        sleep_mock = mocker.patch("openextract._extract.time.sleep")
        once = mocker.patch(
            "openextract._extract._extract_once",
            side_effect=ModelError("upstream down"),
        )

        with pytest.raises(ModelError, match="upstream down"):
            extract(schema=_Person, model="openai:gpt-5", input_file="ignored")

        assert once.call_count == 1
        sleep_mock.assert_not_called()

    def test_retry_succeeds_after_transient_model_errors(self, mocker):
        sleep_mock = mocker.patch("openextract._extract.time.sleep")
        expected = _Person(name="Grace", age=85)
        once = mocker.patch(
            "openextract._extract._extract_once",
            side_effect=[ModelError("flaky"), ModelError("flaky"), expected],
        )

        result = extract(
            schema=_Person,
            model="openai:gpt-5",
            input_file="ignored",
            max_retries=2,
        )

        assert result is expected
        assert once.call_count == 3
        assert sleep_mock.call_count == 2

    def test_retry_exhausted_raises_last_model_error(self, mocker):
        sleep_mock = mocker.patch("openextract._extract.time.sleep")
        once = mocker.patch(
            "openextract._extract._extract_once",
            side_effect=ModelError("persistent"),
        )

        with pytest.raises(ModelError, match="persistent"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=2,
            )

        assert once.call_count == 3
        assert sleep_mock.call_count == 2

    def test_backoff_schedule_uses_exponential_jitter(self, mocker):
        sleep_mock = mocker.patch("openextract._extract.time.sleep")
        mocker.patch(
            "openextract._extract._extract_once",
            side_effect=ModelError("nope"),
        )

        with pytest.raises(ModelError):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=3,
                retry_backoff=1.0,
            )

        delays = [call.args[0] for call in sleep_mock.call_args_list]
        assert len(delays) == 3
        assert delays == sorted(delays)
        assert 1.0 <= delays[0] <= 1.25
        assert 2.0 <= delays[1] <= 2.5
        assert 4.0 <= delays[2] <= 5.0

    def test_schema_validation_error_is_not_retried(self, mocker):
        sleep_mock = mocker.patch("openextract._extract.time.sleep")
        once = mocker.patch(
            "openextract._extract._extract_once",
            side_effect=SchemaValidationError("bad shape"),
        )

        with pytest.raises(SchemaValidationError, match="bad shape"):
            extract(
                schema=_Person,
                model="openai:gpt-5",
                input_file="ignored",
                max_retries=3,
            )

        assert once.call_count == 1
        sleep_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Async helpers and shared mock builder
# ---------------------------------------------------------------------------


def _make_async_agent_mock(mocker, output=None, run_side_effect=None):
    """Patch openextract._extract.Agent and stub the async ``run`` method."""
    agent_instance = MagicMock()
    if run_side_effect is not None:
        agent_instance.run = AsyncMock(side_effect=run_side_effect)
    else:
        run_result = MagicMock()
        run_result.output = output
        agent_instance.run = AsyncMock(return_value=run_result)
    agent_cls = mocker.patch("openextract._extract.Agent", return_value=agent_instance)
    return agent_cls, agent_instance


# ---------------------------------------------------------------------------
# extract_async
# ---------------------------------------------------------------------------


class TestExtractAsync:
    async def test_returns_schema_instance_from_local_file(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        expected = _Person(name="Grace", age=85)
        agent_cls, agent_instance = _make_async_agent_mock(mocker, output=expected)

        result = await extract_async(
            schema=_Person,
            model="openai:gpt-5",
            input_file=str(local),
            instructions="extract",
        )

        assert result is expected
        agent_cls.assert_called_once()
        agent_instance.run.assert_awaited_once()

    async def test_ollama_model_wraps_output_in_native_output(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        expected = _Person(name="Linus", age=54)
        agent_cls, _ = _make_async_agent_mock(mocker, output=expected)

        result = await extract_async(
            schema=_Person,
            model="ollama:llama3",
            input_file=str(local),
        )

        assert result is expected
        output_type = agent_cls.call_args.kwargs["output_type"]
        assert output_type is not _Person
        assert type(output_type).__name__ == "NativeOutput"

    async def test_http_status_error_is_wrapped(self, mocker):
        response = MagicMock()
        response.status_code = 502
        err = httpx.HTTPStatusError("boom", request=MagicMock(), response=response)
        mocker.patch("openextract._extract._get_media", side_effect=err)

        with pytest.raises(UrlFetchError, match="502"):
            await extract_async(schema=_Person, model="openai:gpt-5", input_file="https://x/y")

    async def test_request_error_is_wrapped(self, mocker):
        err = httpx.ConnectError("dns failure")
        mocker.patch("openextract._extract._get_media", side_effect=err)

        with pytest.raises(UrlFetchError, match="dns failure"):
            await extract_async(schema=_Person, model="openai:gpt-5", input_file="https://x/y")

    async def test_validation_error_is_wrapped(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        try:
            _Person(name="x", age="not-an-int")  # type: ignore[arg-type]
        except ValidationError as exc:
            validation_error = exc
        _make_async_agent_mock(mocker, run_side_effect=validation_error)

        with pytest.raises(SchemaValidationError, match="Model output did not match schema"):
            await extract_async(schema=_Person, model="openai:gpt-5", input_file=str(local))

    async def test_generic_exception_is_wrapped_as_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        _make_async_agent_mock(mocker, run_side_effect=RuntimeError("kaboom"))

        with pytest.raises(ExtractionError, match="Extraction failed: kaboom"):
            await extract_async(schema=_Person, model="openai:gpt-5", input_file=str(local))

    async def test_model_keyword_exception_is_not_promoted_to_model_error(self, tmp_path, mocker):
        """Mirrors the sync extract() behavior: substring 'model' no longer triggers ModelError."""
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        _make_async_agent_mock(mocker, run_side_effect=RuntimeError("unknown model identifier"))

        with pytest.raises(ExtractionError) as exc_info:
            await extract_async(schema=_Person, model="openai:gpt-5", input_file=str(local))
        assert not isinstance(exc_info.value, ModelError)

    async def test_missing_media_type_for_bytes_input_raises_type_error(self, mocker):
        """TypeError from _get_media must propagate out of extract_async untouched."""
        _make_async_agent_mock(mocker, output=_Person(name="x", age=1))

        with pytest.raises(TypeError, match="media_type is required"):
            await extract_async(schema=_Person, model="openai:gpt-5", input_file=b"abc")

    async def test_passes_through_existing_extraction_error(self, tmp_path, mocker):
        local = tmp_path / "input.txt"
        local.write_bytes(b"hi")
        original = SchemaValidationError("already mapped")
        _make_async_agent_mock(mocker, run_side_effect=original)

        with pytest.raises(SchemaValidationError) as exc_info:
            await extract_async(schema=_Person, model="openai:gpt-5", input_file=str(local))
        assert exc_info.value is original


# ---------------------------------------------------------------------------
# extract_many
# ---------------------------------------------------------------------------


class TestExtractMany:
    def test_preserves_input_order(self, tmp_path, mocker):
        files = []
        for i in range(4):
            p = tmp_path / f"f{i}.txt"
            p.write_bytes(b"x")
            files.append(str(p))

        people = [_Person(name=f"n{i}", age=i) for i in range(4)]
        # Map each path to its expected person via side_effect.
        path_to_person = dict(zip(files, people, strict=True))

        async def fake_extract_async(schema, model, input_file, instructions=None):
            # Tiny await yields control so tasks can interleave.
            await asyncio.sleep(0)
            return path_to_person[input_file]

        mocker.patch("openextract._extract.extract_async", side_effect=fake_extract_async)

        results = extract_many(schema=_Person, model="openai:gpt-5", input_files=files)

        assert results == people

    def test_max_concurrency_is_respected(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(10)]
        for f in files:
            from pathlib import Path

            Path(f).write_bytes(b"x")

        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def fake_extract_async(schema, model, input_file, instructions=None):
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            return _Person(name=input_file, age=1)

        mocker.patch("openextract._extract.extract_async", side_effect=fake_extract_async)

        results = extract_many(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
            max_concurrency=3,
        )

        assert len(results) == 10
        assert peak <= 3
        assert peak >= 1

    def test_fail_fast_propagates_first_error(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(3)]

        async def fake_extract_async(schema, model, input_file, instructions=None):
            if input_file.endswith("f1.txt"):
                raise ModelError("boom on f1")
            await asyncio.sleep(0.01)
            return _Person(name=input_file, age=1)

        mocker.patch("openextract._extract.extract_async", side_effect=fake_extract_async)

        with pytest.raises(ModelError, match="boom on f1"):
            extract_many(schema=_Person, model="openai:gpt-5", input_files=files)

    def test_return_exceptions_yields_mixed_list(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(3)]

        async def fake_extract_async(schema, model, input_file, instructions=None):
            if input_file.endswith("f1.txt"):
                raise ModelError("boom on f1")
            return _Person(name=input_file, age=1)

        mocker.patch("openextract._extract.extract_async", side_effect=fake_extract_async)

        results = extract_many(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
            return_exceptions=True,
        )

        assert isinstance(results[0], _Person)
        assert isinstance(results[1], ModelError)
        assert isinstance(results[2], _Person)


# ---------------------------------------------------------------------------
# extract_many_async
# ---------------------------------------------------------------------------


class TestExtractManyAsync:
    async def test_fail_fast_propagates_first_error(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(3)]

        async def fake_extract_async(schema, model, input_file, instructions=None):
            if input_file.endswith("f0.txt"):
                raise ModelError("boom first")
            await asyncio.sleep(0.01)
            return _Person(name=input_file, age=1)

        mocker.patch("openextract._extract.extract_async", side_effect=fake_extract_async)

        with pytest.raises(ModelError, match="boom first"):
            await extract_many_async(
                schema=_Person,
                model="openai:gpt-5",
                input_files=files,
            )

    async def test_return_exceptions_yields_mixed_list(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(3)]

        async def fake_extract_async(schema, model, input_file, instructions=None):
            if input_file.endswith("f1.txt"):
                raise SchemaValidationError("bad schema")
            return _Person(name=input_file, age=1)

        mocker.patch("openextract._extract.extract_async", side_effect=fake_extract_async)

        results = await extract_many_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
            return_exceptions=True,
        )

        assert isinstance(results[0], _Person)
        assert isinstance(results[1], SchemaValidationError)
        assert isinstance(results[2], _Person)

    async def test_preserves_input_order(self, tmp_path, mocker):
        files = [str(tmp_path / f"f{i}.txt") for i in range(5)]
        expected = [_Person(name=f, age=i) for i, f in enumerate(files)]
        mapping = dict(zip(files, expected, strict=True))

        async def fake_extract_async(schema, model, input_file, instructions=None):
            await asyncio.sleep(0)
            return mapping[input_file]

        mocker.patch("openextract._extract.extract_async", side_effect=fake_extract_async)

        results = await extract_many_async(
            schema=_Person,
            model="openai:gpt-5",
            input_files=files,
        )

        assert results == expected
