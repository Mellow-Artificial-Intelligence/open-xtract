"""Local PDF parse and citation grounding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from openextract import Citation
from openextract._parse import (
    ParsedDocument,
    ParsedPage,
    ParsedSpan,
    _bbox_for_quote,
    _char_box,
    _char_text,
    _citation_from_value,
    _find_in_text,
    _flush_word,
    _ground_one,
    _iter_field_values,
    _normalized_coco,
    _page_size,
    _page_text,
    _parse_pdf_page,
    _pdf_box_to_coco,
    _union_coco,
    _union_pdf_boxes,
    _walk_fields,
    _word_spans,
    find_span,
    ground_citations,
    maybe_parsed_inputs,
    parsed_run_inputs,
    try_parse_document,
)
from tests.pdf_fixture import synthetic_pdf


def _page(
    text: str,
    *spans: ParsedSpan,
    page: int = 1,
) -> ParsedPage:
    return ParsedPage(page=page, text=text, width=200, height=200, spans=spans)


def test_synthetic_pdf_attaches_parser_box_and_omits_unmatched():
    data = synthetic_pdf("Acme Corp")
    parsed = try_parse_document(data, "application/pdf")
    assert parsed is not None
    assert parsed.has_text()
    assert "Acme" in parsed.as_prompt_text()
    page, bbox = find_span(parsed, "Acme Corp")
    assert page == 1
    assert bbox is not None
    assert all(0 <= value <= 1 for value in bbox)
    assert bbox[2] > 0 and bbox[3] > 0

    cited = ground_citations(
        (
            Citation("vendor", "Acme Corp", page=99, bbox=(0.9, 0.9, 0.05, 0.05)),
            Citation("missing", "no-such-span-in-doc", page=1),
        ),
        parsed,
    )
    assert cited[0].page == 1
    assert cited[0].bbox == bbox
    assert cited[1].page == 1
    assert cited[1].bbox is None


def test_grounding_stamps_page_and_backfills_field_value():
    parsed = ParsedDocument(
        pages=(
            _page(
                "Total 12.50",
                ParsedSpan("Total", 1, (0.1, 0.1, 0.2, 0.05)),
                ParsedSpan("12.50", 1, (0.4, 0.1, 0.2, 0.05)),
            ),
        )
    )
    grounded = ground_citations(
        (),
        parsed,
        {
            "ghost": "zzz-not-in-doc",
            "total": "12.50",
            "label": "Total",
            "empty": None,
            "flag": True,
        },
    )
    fields = {item.field: item for item in grounded}
    assert fields["total"].page == 1
    assert fields["total"].bbox == pytest.approx((0.4, 0.1, 0.2, 0.05))
    assert fields["label"].quote == "Total"
    assert fields["label"].bbox == pytest.approx((0.1, 0.1, 0.2, 0.05))


def test_maybe_parsed_inputs_swaps_in_page_text():
    data = synthetic_pdf("Acme Corp")
    inputs, parsed = maybe_parsed_inputs(data, "application/pdf", parse=True)
    assert parsed is not None
    assert inputs is not None
    assert inputs == parsed_run_inputs(parsed)
    assert "--- Page 1 ---" in inputs[1]
    none_inputs, none_parsed = maybe_parsed_inputs(data, "application/pdf", parse=False)
    assert none_inputs is None and none_parsed is None
    assert maybe_parsed_inputs(b"hello", "text/plain", parse=True) == (None, None)


def test_try_parse_rejects_invalid_and_non_pdf(monkeypatch):
    assert try_parse_document(b"not-a-pdf", "text/plain") is None
    assert try_parse_document(b"%PDF-not-really", "application/pdf") is None
    assert try_parse_document(b"%PDF-magic", None) is None

    def _boom(name, *args, **kwargs):
        if name == "pypdfium2":
            raise ImportError("missing extra")
        return orig_import(name, *args, **kwargs)

    import builtins

    orig_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _boom)
    assert try_parse_document(synthetic_pdf(), "application/pdf") is None


def test_page_helpers_cover_api_fallbacks():
    sized = SimpleNamespace(get_size=lambda: (100, 50))
    assert _page_size(sized) == (100.0, 50.0)
    wide = SimpleNamespace(get_width=lambda: 10, get_height=lambda: 20)
    assert _page_size(wide) == (10.0, 20.0)

    assert _page_text(SimpleNamespace()) == ""
    assert _page_text(SimpleNamespace(get_text_bounded=lambda: "  ")) == ""
    bounded = SimpleNamespace(get_text_bounded=lambda: "Hi", get_text_range=lambda: "x")
    assert _page_text(bounded) == "Hi"
    assert _page_text(SimpleNamespace(get_text_range=lambda: "Range")) == "Range"
    assert _page_text(SimpleNamespace(get_text_range=lambda: None)) == ""

    class _Range:
        def get_text_range(self, *args, **kwargs):
            if args == (0, 1):
                raise TypeError("kwargs only")
            if kwargs == {"index": 0, "count": 1}:
                return "A"
            raise TypeError("nope")

    assert _char_text(_Range(), 0) == "A"
    assert _char_text(SimpleNamespace(), 0) == ""

    class _KwargsOnly:
        def get_text_range(self, **kwargs):
            raise TypeError("still no")

    assert _char_text(_KwargsOnly(), 0) == ""
    assert _char_box(SimpleNamespace(), 0) is None
    assert _char_box(SimpleNamespace(get_charbox=lambda _i: None), 0) is None
    assert _char_box(SimpleNamespace(get_charbox=lambda _i: (1, 2)), 0) is None

    def _raise(_i):
        raise RuntimeError("box")

    assert _char_box(SimpleNamespace(get_charbox=_raise), 0) is None
    assert _char_box(SimpleNamespace(get_charbox=lambda _i: (1.0, 2.0, 3.0, 4.0)), 0) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )


def test_box_helpers_drop_invalid_geometry():
    assert _flush_word([], [], 1, 10, 10) is None
    assert _union_pdf_boxes([], 10, 10) is None
    assert _union_pdf_boxes([(0, 0, 1, 1)], 0, 10) is None
    assert _pdf_box_to_coco(0, 0, 10, 10, 10, 10) == (0.0, 0.0, 1.0, 1.0)
    assert _normalized_coco(0.1, 0.1, 0, 0.1) is None
    assert _normalized_coco(-0.1, 0.1, 0.1, 0.1) is None
    assert _normalized_coco(0.8, 0.1, 0.3, 0.1) is None
    assert _union_coco([]) is None
    assert _union_coco([(0.1, 0.2, 0.1, 0.1), (0.2, 0.25, 0.1, 0.1)]) == pytest.approx(
        (0.1, 0.2, 0.2, 0.15)
    )


def test_find_span_exact_fuzzy_and_hint():
    page = _page(
        "Quarterly revenue beat",
        ParsedSpan("Quarterly", 1, (0.1, 0.1, 0.2, 0.05)),
        ParsedSpan("revenue", 1, (0.3, 0.1, 0.2, 0.05)),
        ParsedSpan("beat", 1, (0.6, 0.1, 0.1, 0.05)),
        page=2,
    )
    other = _page("unrelated", page=1)
    parsed = ParsedDocument(pages=(other, page))
    assert find_span(parsed, None) == (None, None)
    assert find_span(parsed, "   ")[0] is None
    loc_page, bbox = find_span(parsed, "revenue beat", hinted_page=2)
    assert loc_page == 2
    assert bbox is not None
    fuzzy_page, _bbox = find_span(parsed, "qarterly revene")
    assert fuzzy_page == 2
    assert _find_in_text("abc", "x") is None
    assert _find_in_text("abc", "z") is None
    assert find_span(parsed, "zzz", hinted_page=2) == (2, None)


def test_ground_citations_without_parse_drops_model_bbox():
    cited = ground_citations((Citation("vendor", "Acme", 1, (0.1, 0.2, 0.3, 0.04)),), None)
    assert cited == (Citation("vendor", "Acme", 1, None),)


def test_parse_pdf_page_error_is_none(monkeypatch):
    pypdfium2 = pytest.importorskip("pypdfium2")

    class _Boom:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            raise RuntimeError("page")

        def close(self):
            return None

    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _Boom())
    assert try_parse_document(b"%PDF-1.1", "application/x-pdf") is None

    class _Empty:
        def __len__(self):
            return 0

        def close(self):
            return None

    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _Empty())
    assert try_parse_document(b"%PDF-1.1", "application/pdf") is None

    class _Chars:
        def count_chars(self):
            return 4

        def get_text_range(self, index, count=1):
            return "A B "[index]

        def get_charbox(self, index):
            return (float(index), 0.0, float(index + 1), 1.0)

    words = list(_word_spans(_Chars(), 1, 10, 10))
    assert [word.text for word in words] == ["A", "B"]
    assert _flush_word(["Z"], [], 1, 10, 10).bbox is None


def test_walk_nested_fields_and_short_quote():
    parsed = ParsedDocument(
        pages=(
            _page(
                "Ada 36 qty 1",
                ParsedSpan("Ada", 1, (0.1, 0.1, 0.1, 0.1)),
                ParsedSpan("1", 1, (0.5, 0.1, 0.05, 0.05)),
            ),
        )
    )
    grounded = ground_citations(
        (Citation("name", "A", page=1),),
        parsed,
        {"lines": [{"qty": 1}], "skip": [], "name": "Ada"},
    )
    assert grounded[0].page == 1
    assert any(item.field == "lines[0].qty" for item in grounded)

    class _Out(BaseModel):
        vendor: str

    assert list(_iter_field_values(_Out(vendor="Ada"))) == [("vendor", "Ada")]
    assert list(_iter_field_values("nope")) == []
    assert list(_walk_fields({1: "x"}, "")) == []
    assert list(_walk_fields("   ", "blank")) == []
    already = ground_citations((Citation("total", "12.50", 1),), parsed, {"total": "12.50"})
    assert len(already) == 1
    assert _citation_from_value("missing", "zzz", parsed) is None
    none_page = _ground_one(Citation("x", "zzz"), parsed)
    assert none_page.page is None and none_page.bbox is None
    assert _find_in_text("abc", "   ") is None
    assert _bbox_for_quote(_page("Ada"), "   ") is None
    spaced = _page(
        "Acme Corp",
        ParsedSpan("  ", 1, None),
        ParsedSpan("Acme", 1, None),
        ParsedSpan("Other", 1, (0.1, 0.1, 0.1, 0.1)),
    )
    assert _bbox_for_quote(spaced, "Acme") is None
    fuzzy_page = _page(
        "zzz Acme",
        ParsedSpan("zzz", 1, None),
        ParsedSpan("Acme", 1, (0.2, 0.2, 0.2, 0.1)),
    )
    assert _bbox_for_quote(fuzzy_page, "Acmee") == pytest.approx((0.2, 0.2, 0.2, 0.1))
    assert list(_walk_fields(object(), "x")) == []


def test_word_span_space_and_missing_box():
    class _Chars:
        def count_chars(self):
            return 5

        def get_text_range(self, index, count=1):
            return "A  B "[index]

        def get_charbox(self, index):
            return None

    words = list(_word_spans(_Chars(), 1, 10, 10))
    assert [word.text for word in words] == ["A", "B"]

    class _Page:
        def get_size(self):
            return 10, 10

        def get_textpage(self):
            return SimpleNamespace(
                get_text_bounded=lambda: "Hi",
                count_chars=lambda: 0,
            )

    parsed_page = _parse_pdf_page([_Page()], 0)
    assert parsed_page.text == "Hi"


def test_pdf_close_optional(monkeypatch):
    pypdfium2 = pytest.importorskip("pypdfium2")

    class _NoClose:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            raise RuntimeError("page")

    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _NoClose())
    assert try_parse_document(b"%PDF-1.1", "application/pdf") is None

    class _TextPage:
        def get_text_bounded(self):
            return "Hi"

        def count_chars(self):
            return 0

    class _Page:
        def get_size(self):
            return 10, 10

        def get_textpage(self):
            return _TextPage()

    class _Doc:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return _Page()

    monkeypatch.setattr(pypdfium2, "PdfDocument", lambda _data: _Doc())
    parsed = try_parse_document(b"%PDF-1.1", "application/pdf")
    assert parsed is not None
    assert parsed.has_text()
