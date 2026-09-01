"""Opt-in per-field provenance for extraction results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, cast

from pydantic import BaseModel, Field, create_model

from ._types import Citation, T

_MAX_QUOTE = 2000
_FIELD_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*$")
_CREDENTIAL_URL = re.compile(r"https?://[^\s/]*:[^\s/]*@[^\s]+", re.IGNORECASE)
_DATA_URI = re.compile(r"data:[^\s]+", re.IGNORECASE)

CITATION_INSTRUCTIONS = (
    "For every extracted field, cite the source evidence that supports it. "
    "Return citations as a list of objects with field and any of quote, page, bbox. "
    "field is a dotted path matching the schema (for example vendor or lines[0].qty). "
    "quote is the exact text span from the source; never invent text. "
    "page is the 1-indexed page number when the source is paginated. "
    "bbox is optional: four normalized page-relative COCO numbers "
    "[x, y, width, height] in the unit interval, only if you can see the "
    "exact location of the quoted span. Omit bbox when you cannot locate the "
    "span precisely. Never guess or invent a box. "
    "Do not include raw file bytes, URLs, credentials, or file paths. "
    "If a field has no supporting evidence, omit it from citations."
)


class CitationDraft(BaseModel):
    """Structured-output shape the model fills when ``cite=True``."""

    field: str
    quote: str | None = None
    page: int | None = None
    bbox: list[float] | None = None


_WRAP_CACHE: dict[type[BaseModel], type[BaseModel]] = {}


def cited_output_schema(schema: type[BaseModel]) -> type[BaseModel]:
    """Wrap ``schema`` so the model also returns a ``citations`` list."""
    cached = _WRAP_CACHE.get(schema)
    if cached is not None:
        return cached
    wrapped = create_model(
        f"{schema.__name__}WithCitations",
        __module__="openextract._citations",
        output=(schema, ...),
        citations=(list[CitationDraft], Field(default_factory=list)),
    )
    _WRAP_CACHE[schema] = wrapped
    return wrapped


def json_schema_with_citations(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON Schema with the same ``output`` / ``citations`` envelope."""
    return {
        "type": "object",
        "properties": {
            "output": schema,
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "quote": {"type": "string"},
                        "page": {"type": "integer"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": ["field"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["output"],
        "additionalProperties": False,
    }


def with_citation_instructions(instructions: str | None) -> str:
    """Append citation guidance without dropping caller instructions."""
    if instructions and instructions.strip():
        return f"{instructions.strip()}\n\n{CITATION_INSTRUCTIONS}"
    return CITATION_INSTRUCTIONS


def prepare_cited_run(
    schema: type[T],
    instructions: str | None,
    cite: bool,
) -> tuple[type[BaseModel], str | None]:
    """Return the output type and instructions for one extraction run."""
    if not cite:
        return schema, instructions
    return cited_output_schema(schema), with_citation_instructions(instructions)


def split_cited_output(
    raw: object,
    schema: type[T],
    *,
    cite: bool,
) -> tuple[T, tuple[Citation, ...]]:
    """Unwrap a cited model payload into ``(schema instance, citations)``.

    When ``cite`` is false the raw output is returned unchanged and citations
    are empty, matching the default extract path.
    """
    if not cite:
        return cast(T, raw), ()
    wrapper_type = cited_output_schema(schema)
    wrapper = cast(
        Any, raw if isinstance(raw, wrapper_type) else wrapper_type.model_validate(raw)
    )
    return cast(T, wrapper.output), citations_from_payload(wrapper.citations)


def citations_from_payload(payload: object) -> tuple[Citation, ...]:
    """Sanitize a model or JSON citation list into public :class:`Citation` values."""
    if not isinstance(payload, Sequence) or isinstance(payload, str | bytes):
        return ()
    out: list[Citation] = []
    for item in payload:
        citation = sanitize_citation(item)
        if citation is not None:
            out.append(citation)
    return tuple(out)


def sanitize_citation(draft: object) -> Citation | None:
    """Drop unsafe or unusable citation payloads; never raise on a bad item."""
    if isinstance(draft, Citation | CitationDraft):
        field, quote, page, bbox = draft.field, draft.quote, draft.page, draft.bbox
    elif isinstance(draft, dict):
        payload = cast(dict[str, Any], draft)
        field = payload.get("field") or payload.get("field_path")
        quote = payload.get("quote", payload.get("reference_text"))
        page = payload.get("page")
        bbox = payload.get("bbox")
    else:
        return None
    if not isinstance(field, str) or not _FIELD_PATH.fullmatch(field):
        return None
    if isinstance(quote, str):
        quote = _sanitize_quote(quote) or None
    elif quote is not None:
        quote = None
    if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
        page = None
    bbox = _sanitize_bbox(bbox)
    if quote is None and page is None:
        return None
    return Citation(field=field, quote=quote, page=page, bbox=bbox)


def _sanitize_quote(quote: str) -> str:
    """Collapse whitespace and strip credential-bearing URLs and data URIs."""
    text = " ".join(quote.split())
    text = _CREDENTIAL_URL.sub("[redacted]", text)
    text = _DATA_URI.sub("[redacted]", text)
    if len(text) > _MAX_QUOTE:
        text = text[:_MAX_QUOTE]
    return text


def _sanitize_bbox(bbox: object) -> tuple[float, float, float, float] | None:
    """Keep a model-supplied normalized COCO box; never invent or rescale one."""
    if isinstance(bbox, tuple):
        bbox = list(bbox)
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    values: list[float] = []
    for item in bbox:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return None
        values.append(float(item))
    x, y, width, height = values
    # ExtractBench boxes are page-normalized. Pixel-space numbers cannot be
    # converted without page dimensions, so drop them rather than guess.
    if width <= 0 or height <= 0:
        return None
    if any(value < 0 or value > 1 for value in (x, y, width, height)):
        return None
    if x + width > 1.0001 or y + height > 1.0001:
        return None
    return (x, y, width, height)


def field_citations_for_extractbench(
    citations: Iterable[Citation],
) -> list[dict[str, Any]]:
    """Map library citations to ExtractBench ``FieldCitation`` payloads."""
    return [payload for citation in citations if (payload := citation.as_field_citation())]
