"""Out-of-the-box extraction styles backed by the Pydantic AI harness."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

from .exceptions import ProviderNotInstalledError

_CODE_VIRTUAL_ROOT = "/work"
_BINARY_MEDIA_PREFIXES = ("image/", "audio/", "video/")
_BINARY_MEDIA_TYPES = frozenset(
    {
        "application/pdf",
        "application/zip",
        "application/gzip",
        "application/x-gzip",
        "application/x-tar",
    }
)
_TEXT_APPLICATION_TYPES = frozenset(
    {
        "application/csv",
        "application/graphql",
        "application/javascript",
        "application/json",
        "application/ld+json",
        "application/sql",
        "application/toml",
        "application/xml",
        "application/x-ndjson",
        "application/x-sh",
        "application/x-yaml",
        "application/yaml",
    }
)
_DOCUMENT_FILENAMES = {
    "csv": "document.csv",
    "html": "document.html",
    "javascript": "document.js",
    "json": "document.json",
    "ld+json": "document.json",
    "markdown": "document.md",
    "plain": "document.txt",
    "tab-separated-values": "document.tsv",
    "toml": "document.toml",
    "x-markdown": "document.md",
    "x-ndjson": "document.ndjson",
    "x-sh": "document.sh",
    "x-yaml": "document.yaml",
    "xml": "document.xml",
    "yaml": "document.yaml",
}


class ExtractionStyle(StrEnum):
    """How an extraction run inspects the input.

    ``direct``
        Pass the media bytes to the model in one shot (default).
    ``search``
        For text, give the model sandboxed file tools (read, regex search,
        glob) against a workspace copy of the document.
    ``code``
        For text, give the model a sandboxed ``run_code`` tool that can open
        and parse the document with Python.
    """

    DIRECT = "direct"
    SEARCH = "search"
    CODE = "code"


def normalize_style(style: ExtractionStyle | str) -> ExtractionStyle:
    """Return a valid :class:`ExtractionStyle` or raise ``ValueError``."""
    try:
        return ExtractionStyle(style)
    except ValueError:
        allowed = ", ".join(repr(item.value) for item in ExtractionStyle)
        raise ValueError(f"style must be one of {allowed}; got {style!r}.") from None


def _bare_media_type(media_type: str) -> str:
    return media_type.split(";", 1)[0].strip().lower()


def is_text_media_type(media_type: str) -> bool:
    """Return whether ``media_type`` is a known textual MIME type."""
    bare = _bare_media_type(media_type)
    return (
        bare.startswith("text/")
        or bare in _TEXT_APPLICATION_TYPES
        or bare.endswith(("+json", "+xml", "+yaml"))
    )


def is_binary_media_type(media_type: str) -> bool:
    """Return whether ``media_type`` is a known non-text MIME type."""
    bare = _bare_media_type(media_type)
    return bare.startswith(_BINARY_MEDIA_PREFIXES) or bare in _BINARY_MEDIA_TYPES


def document_filename(media_type: str) -> str:
    """Return a stable workspace filename for a textual media type."""
    subtype = _bare_media_type(media_type).rsplit("/", 1)[-1]
    return _DOCUMENT_FILENAMES.get(subtype, "document.txt")


def decode_text_document(data: bytes, media_type: str, *, style: ExtractionStyle) -> str:
    """Decode UTF-8 text for search/code styles; reject binary inputs."""
    if is_binary_media_type(media_type) and not is_text_media_type(media_type):
        raise ValueError(
            f"style {style.value!r} requires a text document; got media_type={media_type!r}. "
            "Use style='direct' for PDFs, images, audio, and video."
        )
    if b"\x00" in data:
        raise ValueError(
            f"style {style.value!r} requires a text document; the input contains NUL bytes. "
            "Use style='direct' instead."
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"style {style.value!r} requires UTF-8 text; the input is not valid UTF-8. "
            "Use style='direct' instead."
        ) from exc


def materialize_text_document(
    workspace: Path,
    data: bytes,
    media_type: str,
    *,
    style: ExtractionStyle,
) -> str:
    """Write the decoded document into ``workspace`` and return its filename."""
    filename = document_filename(media_type)
    (workspace / filename).write_text(
        decode_text_document(data, media_type, style=style),
        encoding="utf-8",
    )
    return filename


def _search_capabilities(workspace: Path) -> list[object]:
    try:
        filesystem = __import__("pydantic_ai_harness", fromlist=["FileSystem"]).FileSystem
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "style='search' requires pydantic-ai-harness. "
            "Install it with: pip install pydantic-ai-harness "
            f"(or 'pip install pydantic-ai-harness[codemode]'). Original error: {exc}"
        ) from exc
    return [filesystem(root_dir=workspace)]


def _code_capabilities(workspace: Path) -> list[object]:
    try:
        harness = __import__("pydantic_ai_harness", fromlist=["CodeMode"])
        monty = __import__("pydantic_monty", fromlist=["MountDir"])
    except ImportError as exc:
        raise ProviderNotInstalledError(
            "style='code' requires pydantic-ai-harness with the code-mode extra. "
            "Install it with: pip install 'pydantic-ai-harness[codemode]' "
            f"(or 'pip install pydantic-ai-harness[code-mode]'). Original error: {exc}"
        ) from exc
    return [
        harness.CodeMode(
            mount=monty.MountDir(
                virtual_path=_CODE_VIRTUAL_ROOT,
                host_path=workspace,
                mode="read-only",
            )
        )
    ]


def style_capabilities(style: ExtractionStyle, workspace: Path) -> list[object]:
    """Return Pydantic AI capabilities that implement ``style``."""
    if style is ExtractionStyle.SEARCH:
        return _search_capabilities(workspace)
    if style is ExtractionStyle.CODE:
        return _code_capabilities(workspace)
    return []


def style_run_inputs(style: ExtractionStyle, filename: str) -> list[str]:
    """Return the user prompt for a search or code extraction."""
    if style is ExtractionStyle.SEARCH:
        return [
            "Extract the requested information from the document "
            f"{filename!r} in the workspace. Use search_files, read_file, "
            "find_files, list_directory, and file_info to inspect it. Search "
            "before reading large files; do not assume unseen contents."
        ]
    return [
        "Extract the requested information by writing Python against "
        f"{_CODE_VIRTUAL_ROOT}/{filename}. Read the file with pathlib or "
        "open(), then parse, filter, and compute the structured result."
    ]


@contextmanager
def prepared_style_run(
    style: ExtractionStyle,
    file_bytes: bytes,
    file_type: str,
) -> Iterator[tuple[list[object], list[str] | None]]:
    """Yield ``(extra_capabilities, run_inputs)`` for one extraction.

    ``direct`` yields no extra capabilities and ``None`` inputs so the caller
    can pass media as ``BinaryContent``. ``search`` and ``code`` materialize a
    UTF-8 workspace that lives until the context exits, including retries.
    """
    if style is ExtractionStyle.DIRECT:
        yield [], None
        return
    with tempfile.TemporaryDirectory(prefix="openextract-") as tmp:
        workspace = Path(tmp)
        filename = materialize_text_document(workspace, file_bytes, file_type, style=style)
        yield style_capabilities(style, workspace), style_run_inputs(style, filename)
