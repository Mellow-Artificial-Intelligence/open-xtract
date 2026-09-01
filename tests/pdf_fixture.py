"""Tiny PDFs used to prove parser-backed boxes and windowed extract."""

from __future__ import annotations

from collections.abc import Sequence


def synthetic_pdf(
    text: str = "Acme Corp",
    *,
    pages: Sequence[str] | None = None,
    width: int = 200,
    height: int = 200,
    x: int = 24,
    y: int = 160,
) -> bytes:
    """Build a PDF with Helvetica text at a known position on each page."""
    texts = list(pages) if pages is not None else [text]
    if not texts:
        texts = [text]
    page_count = len(texts)
    kids = " ".join(f"{3 + index} 0 R" for index in range(page_count))
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>",
    ]
    content_ids = [3 + page_count + index for index in range(page_count)]
    font_id = 3 + 2 * page_count
    for index in range(page_count):
        content_id = content_ids[index]
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
    for page_text in texts:
        escaped = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content = f"BT /F1 24 Tf {x} {y} Td ({escaped}) Tj ET"
        objects.append(
            f"<< /Length {len(content.encode('latin-1'))} >>\nstream\n{content}\nendstream"
        )
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.1\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{index} 0 obj\n".encode())
        out.extend(body.encode("latin-1"))
        out.extend(b"\nendobj\n")
    xref_at = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
        ).encode()
    )
    return bytes(out)
