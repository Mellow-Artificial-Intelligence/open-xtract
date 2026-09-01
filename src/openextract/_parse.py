"""Local parse-then-extract: page text, page images, and parser-backed boxes."""

from __future__ import annotations

import math
import struct
import threading
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel

from ._types import Citation

_PDF_TYPES = frozenset({"application/pdf", "application/x-pdf"})
_PAGE_HEADER = "--- Page {page} ---"
_FUZZY_RATIO = 0.85
# ~3k tokens of document text. GLM-class context still needs room for a large
# ExtractBench schema, citation envelope, and output. Oversized pages are split.
# Slide decks can sit under the char budget and still blow context as one prompt
# (Veralto was a single 80k window). One page per window forces those decks to
# split; an oversized page is still sliced by the char budget.
DEFAULT_PARSE_WINDOW_CHARS = 12_000
DEFAULT_PARSE_WINDOW_PAGES = 1
# Long-edge cap for scanned page rasters. scale=2 letter PNGs were ~400KB and
# token-limited GLM one page at a time; boxes stay on the PDF page, not pixels.
DEFAULT_RENDER_MAX_EDGE = 768
# pypdfium2 / PDFium is not safe across threads in one process.
_PDFIUM_LOCK = threading.Lock()


@dataclass(frozen=True)
class ParsedSpan:
    """A parser word/span with an optional normalized COCO box."""

    text: str
    page: int
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class ParsedPage:
    """One 1-indexed page of extracted text and word spans."""

    page: int
    text: str
    width: float
    height: float
    spans: tuple[ParsedSpan, ...]
    image: bytes | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Page-indexed local parse used to ground citations."""

    pages: tuple[ParsedPage, ...]

    def has_text(self) -> bool:
        return any(page.text.strip() for page in self.pages)

    def has_images(self) -> bool:
        return any(page.image for page in self.pages)

    def as_prompt_text(self) -> str:
        """Render page markers so the model can cite 1-indexed pages."""
        return _pages_prompt_text(self.pages)


def try_parse_document(data: bytes, media_type: str | None) -> ParsedDocument | None:
    """Parse a paginated document when a local parser can handle it.

    Returns ``None`` when the extra is missing, the type is not a PDF, or the
    bytes are not a readable PDF. Never raises on a bad file.
    """
    if not _is_pdf(data, media_type):
        return None
    return _parse_pdf(data)


def parsed_run_inputs(parsed: ParsedDocument) -> list[str]:
    """Prompt inputs that replace raw PDF bytes when parse text exists."""
    return [
        "Extract the requested information from this document. "
        "Page boundaries are marked as '--- Page N ---' (1-indexed).",
        parsed.as_prompt_text(),
    ]


def parsed_image_inputs(parsed: ParsedDocument) -> list:
    """Prompt inputs that send locally rendered page images, never the PDF.

    ``BinaryContent`` is imported here so ``import openextract`` stays
    pydantic-ai-free (see ``test_package_import_defers_pydantic_ai_runtime``).
    """
    from pydantic_ai import BinaryContent

    parts: list = [
        "Extract the requested information from this document. "
        "Page images follow in 1-indexed order, marked as '--- Page N ---'."
    ]
    for page in parsed.pages:
        parts.append(_PAGE_HEADER.format(page=page.page))
        if page.image:
            parts.append(BinaryContent(data=page.image, media_type="image/png"))
    return parts


def _page_block(page: ParsedPage) -> str:
    return f"{_PAGE_HEADER.format(page=page.page)}\n{page.text}"


def _pages_prompt_text(pages: tuple[ParsedPage, ...]) -> str:
    return "\n\n".join(_page_block(page) for page in pages).strip()


def _pages_prompt_len(pages: tuple[ParsedPage, ...]) -> int:
    if not pages:
        return 0
    return sum(len(_page_block(page)) for page in pages) + 2 * (len(pages) - 1)


def parse_windows(
    parsed: ParsedDocument,
    *,
    max_chars: int | None = None,
    max_pages: int | None = None,
) -> tuple[ParsedDocument, ...]:
    """Split ``parsed`` into page-contiguous windows under ``max_chars``.

    A page larger than the budget is sliced so each model call stays inside
    GLM-class context. Slices keep the original page number and parser spans;
    boxes are still resolved against the full parse, never invented. Decks that
    fit the character budget still split once they exceed ``max_pages``
    (default one page, so a Veralto-class deck cannot stay one prompt). Empty
    documents yield the original parse.
    """
    budget = DEFAULT_PARSE_WINDOW_CHARS if max_chars is None else max_chars
    page_cap = DEFAULT_PARSE_WINDOW_PAGES if max_pages is None else max_pages
    pages = parsed.pages
    if not pages or (_pages_prompt_len(pages) <= budget and len(pages) <= page_cap):
        return (parsed,)
    slices = tuple(slice_page for page in pages for slice_page in _split_page(page, budget))
    windows: list[ParsedDocument] = []
    current: list[ParsedPage] = []
    current_len = 0
    for page in slices:
        block_len = len(_page_block(page))
        extra = block_len if not current else block_len + 2
        if current and (current_len + extra > budget or len(current) >= page_cap):
            windows.append(ParsedDocument(pages=tuple(current)))
            current = [page]
            current_len = block_len
            continue
        current.append(page)
        current_len += extra
    windows.append(ParsedDocument(pages=tuple(current)))
    return tuple(windows)


def _split_page(page: ParsedPage, budget: int) -> tuple[ParsedPage, ...]:
    """Slice one page so each ``--- Page N ---`` block fits ``budget``."""
    block_len = len(_page_block(page))
    if block_len <= budget:
        return (page,)
    header_len = len(_PAGE_HEADER.format(page=page.page)) + 1
    body_budget = max(1, budget - header_len)
    text = page.text
    chunks: list[ParsedPage] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + body_budget)
        if end < len(text):
            break_at = text.rfind("\n", start, end)
            if break_at <= start:
                break_at = text.rfind(" ", start, end)
            if break_at > start:
                end = break_at + 1
        chunks.append(
            ParsedPage(
                page=page.page,
                text=text[start:end],
                width=page.width,
                height=page.height,
                spans=page.spans,
                image=page.image,
            )
        )
        start = end
    return tuple(chunks) or (page,)


def parsed_window_inputs(
    parsed: ParsedDocument | None,
    fallback_inputs: list,
    *,
    max_chars: int | None = None,
    max_pages: int | None = None,
) -> list[list]:
    """Return per-window run inputs, or ``[fallback_inputs]`` for the fast path.

    One window keeps the original inputs object so callers that patch
    ``_extract_once`` still see a single call with the prepared prompt.
    """
    if parsed is None or not parsed.pages:
        return [fallback_inputs]
    if not parsed.has_text():
        windows = parse_windows(parsed, max_chars=max_chars, max_pages=max_pages)
        if parsed.has_images():
            return [parsed_image_inputs(window) for window in windows]
        return [parsed_run_inputs(window) for window in windows]
    windows = parse_windows(parsed, max_chars=max_chars, max_pages=max_pages)
    if len(windows) == 1:
        return [fallback_inputs]
    return [parsed_run_inputs(window) for window in windows]


def maybe_parsed_inputs(
    file_bytes: bytes,
    file_type: str,
    *,
    parse: bool,
) -> tuple[list | None, ParsedDocument | None]:
    """Return page-indexed prompt inputs when a local parse can replace the file.

    Text PDFs become page-marked strings. Scanned / empty-text PDFs become
    locally rendered page images (or page headers if render failed). Never
    returns ``None`` inputs for a paginated parse — that would upload the PDF
    to the provider's document-parse engine.
    """
    parsed = try_parse_document(file_bytes, file_type) if parse else None
    if parsed is None:
        return None, None
    if parsed.has_text():
        return parsed_run_inputs(parsed), parsed
    if parsed.has_images():
        return parsed_image_inputs(parsed), parsed
    if parsed.pages:
        return parsed_run_inputs(parsed), parsed
    return None, parsed


def ground_citations(
    citations: tuple[Citation, ...],
    parsed: ParsedDocument | None,
    output: object | None = None,
) -> tuple[Citation, ...]:
    """Stamp page from the parse index and attach parser boxes only.

    Model-supplied boxes are ignored. A citation with a valid page is kept even
    when the quote is short. Unmatched spans omit ``bbox`` rather than guessing.
    """
    if parsed is None:
        return tuple(
            Citation(field=item.field, quote=item.quote, page=item.page, bbox=None)
            for item in citations
        )
    grounded = [_ground_one(item, parsed) for item in citations]
    seen = {item.field for item in grounded}
    extras: list[Citation] = []
    for field, value in _iter_field_values(output):
        if field in seen:
            continue
        extra = _citation_from_value(field, value, parsed)
        if extra is not None:
            extras.append(extra)
            seen.add(field)
    return tuple(grounded + extras)


def _is_pdf(data: bytes, media_type: str | None) -> bool:
    if media_type in _PDF_TYPES:
        return True
    return data.startswith(b"%PDF")


def _parse_pdf(data: bytes) -> ParsedDocument | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    with _PDFIUM_LOCK:
        try:
            pdf = pdfium.PdfDocument(data)
        except Exception:
            return None
        try:
            pages = tuple(_parse_pdf_page(pdf, index) for index in range(len(pdf)))
        except Exception:
            return None
        finally:
            close = getattr(pdf, "close", None)
            if callable(close):
                close()
        return ParsedDocument(pages=pages) if pages else None


def _parse_pdf_page(pdf: Any, index: int) -> ParsedPage:
    page = pdf[index]
    width, height = _page_size(page)
    textpage = page.get_textpage()
    try:
        text = _page_text(textpage)
        spans = tuple(_word_spans(textpage, index + 1, width, height))
    finally:
        close = getattr(textpage, "close", None)
        if callable(close):
            close()
    image = None if text.strip() else _render_page_png(page)
    return ParsedPage(
        page=index + 1, text=text, width=width, height=height, spans=spans, image=image
    )


def _render_scale(width: float, height: float, max_edge: int = DEFAULT_RENDER_MAX_EDGE) -> float:
    """Scale that fits the page in ``max_edge`` pixels without upscaling."""
    longest = max(width, height)
    if longest <= 0:
        return 1.0
    return min(1.0, max_edge / longest)


def _render_page_png(page: Any) -> bytes | None:
    """Rasterize one PDF page to a compact grayscale PNG.

    Callers must hold ``_PDFIUM_LOCK``.
    """
    render = getattr(page, "render", None)
    if not callable(render):
        return None
    try:
        width, height = _page_size(page)
        scale = _render_scale(width, height)
    except Exception:
        scale = 1.0
    try:
        bitmap = render(scale=scale, rev_byteorder=True)
    except TypeError:
        try:
            bitmap = render(scale=scale)
        except Exception:
            return None
    except Exception:
        return None
    try:
        return _bitmap_to_png(bitmap)
    finally:
        close = getattr(bitmap, "close", None)
        if callable(close):
            close()


def _bitmap_to_png(bitmap: Any) -> bytes | None:
    width = int(getattr(bitmap, "width", 0) or 0)
    height = int(getattr(bitmap, "height", 0) or 0)
    mode = str(getattr(bitmap, "mode", "") or "")
    channels = int(getattr(bitmap, "n_channels", 0) or 0) or _channels_for_mode(mode)
    stride = int(getattr(bitmap, "stride", 0) or 0)
    buffer = getattr(bitmap, "buffer", None)
    if width <= 0 or height <= 0 or buffer is None or channels not in {1, 3, 4}:
        return None
    if stride < width * channels:
        return None
    raw = bytes(buffer)
    row = width * channels
    packed = bytearray(height * row)
    swap = mode.startswith("BGR")
    for y in range(height):
        src = raw[y * stride : y * stride + row]
        dest = y * row
        if swap and channels >= 3:
            for x in range(width):
                i, j = x * channels, dest + x * channels
                packed[j] = src[i + 2]
                packed[j + 1] = src[i + 1]
                packed[j + 2] = src[i]
                if channels == 4:
                    packed[j + 3] = src[i + 3]
        else:
            packed[dest : dest + row] = src
    luma, luma_channels = _luma_pixels(bytes(packed), width, height, channels)
    return _encode_png(width, height, luma, luma_channels)


def _luma_pixels(pixels: bytes, width: int, height: int, channels: int) -> tuple[bytes, int]:
    """Convert packed RGB(A) to 8-bit grayscale. 1-channel input is returned as-is."""
    if channels == 1:
        return pixels, 1
    out = bytearray(width * height)
    for index in range(width * height):
        offset = index * channels
        out[index] = (pixels[offset] * 77 + pixels[offset + 1] * 150 + pixels[offset + 2] * 29) >> 8
    return bytes(out), 1


def _channels_for_mode(mode: str) -> int:
    if mode in {"L", "gray", "GRAY"}:
        return 1
    if mode in {"RGB", "BGR"}:
        return 3
    if mode in {"RGBA", "BGRA", "RGBx", "BGRx"}:
        return 4
    return 0


def _encode_png(width: int, height: int, pixels: bytes, channels: int) -> bytes:
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    raw = bytearray()
    row = width * channels
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * row : (y + 1) * row])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def _page_size(page: Any) -> tuple[float, float]:
    size = getattr(page, "get_size", None)
    if callable(size):
        width, height = size()
        return float(width), float(height)
    return float(page.get_width()), float(page.get_height())


def _page_text(textpage: Any) -> str:
    bounded = getattr(textpage, "get_text_bounded", None)
    if callable(bounded):
        text = bounded()
        if isinstance(text, str) and text.strip():
            return text
    ranged = getattr(textpage, "get_text_range", None)
    if callable(ranged):
        text = ranged()
        if isinstance(text, str):
            return text
    return ""


def _word_spans(textpage: Any, page: int, width: float, height: float) -> Iterator[ParsedSpan]:
    count = int(textpage.count_chars())
    chars: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    for index in range(count):
        char = _char_text(textpage, index)
        box = _char_box(textpage, index)
        if not char or char.isspace():
            span = _flush_word(chars, boxes, page, width, height)
            if span is not None:
                yield span
            chars, boxes = [], []
            continue
        chars.append(char)
        if box is not None:
            boxes.append(box)
    span = _flush_word(chars, boxes, page, width, height)
    if span is not None:
        yield span


def _char_text(textpage: Any, index: int) -> str:
    getter = getattr(textpage, "get_text_range", None)
    if not callable(getter):
        return ""
    try:
        text = getter(index, 1)
    except TypeError:
        try:
            text = getter(index=index, count=1)
        except TypeError:
            return ""
    return text if isinstance(text, str) else ""


def _char_box(textpage: Any, index: int) -> tuple[float, float, float, float] | None:
    getter = getattr(textpage, "get_charbox", None)
    if not callable(getter):
        return None
    try:
        box = getter(index)
    except Exception:
        return None
    if not isinstance(box, tuple | list) or len(box) != 4:
        return None
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def _flush_word(
    chars: list[str],
    boxes: list[tuple[float, float, float, float]],
    page: int,
    width: float,
    height: float,
) -> ParsedSpan | None:
    text = "".join(chars)
    if not text:
        return None
    return ParsedSpan(text=text, page=page, bbox=_union_pdf_boxes(boxes, width, height))


def _union_pdf_boxes(
    boxes: list[tuple[float, float, float, float]],
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    if not boxes or width <= 0 or height <= 0:
        return None
    left = min(box[0] for box in boxes)
    bottom = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    top = max(box[3] for box in boxes)
    return _pdf_box_to_coco(left, bottom, right, top, width, height)


def _pdf_box_to_coco(
    left: float,
    bottom: float,
    right: float,
    top: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    x = left / width
    y = (height - top) / height
    box_width = (right - left) / width
    box_height = (top - bottom) / height
    return _normalized_coco(x, y, box_width, box_height)


def _normalized_coco(
    x: float, y: float, width: float, height: float
) -> tuple[float, float, float, float] | None:
    if width <= 0 or height <= 0:
        return None
    if any(value < 0 or value > 1 for value in (x, y, width, height)):
        return None
    if x + width > 1.0001 or y + height > 1.0001:
        return None
    return (x, y, width, height)


def _ground_one(citation: Citation, parsed: ParsedDocument) -> Citation:
    page, bbox = find_span(parsed, citation.quote, hinted_page=citation.page)
    if page is None:
        page = citation.page if citation.page is not None and citation.page >= 1 else None
    return Citation(field=citation.field, quote=citation.quote, page=page, bbox=bbox)


def find_span(
    parsed: ParsedDocument,
    quote: str | None,
    *,
    field_value: str | None = None,
    hinted_page: int | None = None,
) -> tuple[int | None, tuple[float, float, float, float] | None]:
    """Locate ``quote`` (then ``field_value``) on the parse: exact, then fuzzy."""
    needles = [text for text in (quote, field_value) if text and str(text).strip()]
    if not needles:
        return _valid_page(hinted_page), None
    pages = _pages_for_hint(parsed, hinted_page)
    for needle in needles:
        for page in pages:
            if _haystack_has(page.text, needle):
                return page.page, _bbox_for_quote(page, needle)
    return _valid_page(hinted_page), None


def _pages_for_hint(parsed: ParsedDocument, hinted_page: int | None) -> tuple[ParsedPage, ...]:
    if hinted_page is None:
        return parsed.pages
    hinted = tuple(page for page in parsed.pages if page.page == hinted_page)
    others = tuple(page for page in parsed.pages if page.page != hinted_page)
    return hinted + others


def _valid_page(page: int | None) -> int | None:
    return page if isinstance(page, int) and not isinstance(page, bool) and page >= 1 else None


def _haystack_has(haystack: str, needle: str) -> bool:
    return _find_in_text(haystack, needle) is not None


def _find_in_text(haystack: str, needle: str) -> int | None:
    norm_h = _normalize(haystack)
    norm_n = _normalize(needle)
    if not norm_n:
        return None
    index = norm_h.find(norm_n)
    if index >= 0:
        return index
    if len(norm_n) < 2:
        return None
    window = len(norm_n)
    step = max(1, window // 4)
    best_index: int | None = None
    best_ratio = _FUZZY_RATIO
    limit = max(1, len(norm_h) - window + 1)
    for start in range(0, limit, step):
        ratio = SequenceMatcher(None, norm_h[start : start + window], norm_n).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = start
    return best_index


def _bbox_for_quote(page: ParsedPage, quote: str) -> tuple[float, float, float, float] | None:
    needle = _normalize(quote)
    if not needle:
        return None
    spans = page.spans
    for start in range(len(spans)):
        parts: list[str] = []
        boxes: list[tuple[float, float, float, float]] = []
        for span in spans[start:]:
            token = _normalize(span.text)
            if not token:
                continue
            parts.append(token)
            if span.bbox is not None:
                boxes.append(span.bbox)
            joined = " ".join(parts)
            if joined == needle or needle in joined:
                return _union_coco(boxes)
            if not (needle.startswith(joined) or joined in needle):
                break
    best_box: tuple[float, float, float, float] | None = None
    best_ratio = _FUZZY_RATIO
    for span in spans:
        if span.bbox is None:
            continue
        ratio = SequenceMatcher(None, _normalize(span.text), needle).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_box = span.bbox
    return best_box


def _union_coco(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return _normalized_coco(x0, y0, x1 - x0, y1 - y0)


def _citation_from_value(field: str, value: str, parsed: ParsedDocument) -> Citation | None:
    """Locate ``value`` in the parse, including numeric display variants.

    Window extract often returns schema numbers (``1234.0``) with an empty
    citation list. ExtractBench still needs a page, so try thousands-separated
    and whole-integer forms that appear in the PDF.
    """
    for needle in _value_needles(value):
        page, bbox = find_span(parsed, needle)
        if page is not None:
            return Citation(field=field, quote=needle, page=page, bbox=bbox)
    return None


def _value_needles(value: str) -> tuple[str, ...]:
    """Search texts for a field value, including ``1,234`` / ``1234.0`` forms."""
    texts = [value]
    stripped = value.replace(",", "").replace("$", "").replace("£", "").replace("€", "").strip()
    if stripped and stripped != value:
        texts.append(stripped)
    try:
        number = float(stripped)
    except ValueError:
        return tuple(dict.fromkeys(texts))
    if not math.isfinite(number) or abs(number) >= 1e15:
        return tuple(dict.fromkeys(texts))
    if number.is_integer():
        integer = int(number)
        texts.append(str(integer))
        texts.append(f"{integer:,}")
    else:
        texts.append(f"{number:.2f}")
        texts.append(f"{number:,.2f}")
    return tuple(dict.fromkeys(text for text in texts if text))


def _iter_field_values(output: object) -> Iterator[tuple[str, str]]:
    if output is None:
        return
    data: Any = output.model_dump() if isinstance(output, BaseModel) else output
    if not isinstance(data, dict):
        return
    yield from _walk_fields(data, "")


def _walk_fields(node: object, prefix: str) -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            path = f"{prefix}.{key}" if prefix else key
            yield from _walk_fields(value, path)
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_fields(value, f"{prefix}[{index}]")
        return
    if node is None or isinstance(node, bool):
        return
    if isinstance(node, str | int | float):
        text = str(node).strip()
        if text:
            yield prefix, text


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
