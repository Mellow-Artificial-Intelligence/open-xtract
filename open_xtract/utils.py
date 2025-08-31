import base64
import os
from typing import Optional
from urllib.parse import urlparse

from mistralai import Mistral


def _get_client() -> Mistral:
    """Create a Mistral client using MISTRAL_API_KEY from env."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")
    return Mistral(api_key=api_key)


def _b64encode_file(path: str) -> str:
    """Read a file as bytes and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_image_mime_from_extension(path: str, default: str = "image/jpeg") -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):  # common default
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext in (".tif", ".tiff"):
        return "image/tiff"
    if ext == ".bmp":
        return "image/bmp"
    return default


def ocr(
    source: str,
    *,
    model: str = "mistral-ocr-latest",
    include_image_base64: bool = True,
    default_image_mime: str = "image/jpeg",
) -> dict:
    """
    Run OCR from a single input string.

    Accepts one of:
    - HTTP(S) URL to a PDF or image
    - Local file path to a PDF or image
    - data: URL (application/pdf or image/*)
    """
    client = _get_client()

    # Case 1: data URL provided directly
    if source.startswith("data:"):
        kind = "image_url"
        if source.startswith("data:application/pdf"):
            kind = "document_url"
        document = {"type": kind, ("document_url" if kind == "document_url" else "image_url"): source}
        return client.ocr.process(
            model=model,
            document=document,
            include_image_base64=include_image_base64,
        )

    # Case 2: web URL
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        path_lower = (parsed.path or "").lower()
        is_pdf = path_lower.endswith(".pdf") or ".pdf" in path_lower
        if is_pdf:
            document = {"type": "document_url", "document_url": source}
        else:
            document = {"type": "image_url", "image_url": source}
        return client.ocr.process(
            model=model,
            document=document,
            include_image_base64=include_image_base64,
        )

    # Case 3: local file path
    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".pdf":
            b64 = _b64encode_file(source)
            data_url = f"data:application/pdf;base64,{b64}"
            document = {"type": "document_url", "document_url": data_url}
        else:
            mime = _guess_image_mime_from_extension(source, default=default_image_mime)
            b64 = _b64encode_file(source)
            data_url = f"data:{mime};base64,{b64}"
            document = {"type": "image_url", "image_url": data_url}
        return client.ocr.process(
            model=model,
            document=document,
            include_image_base64=include_image_base64,
        )

    raise ValueError("source must be a data URL, web URL, or existing local file path")


if __name__ == "__main__":
    print(ocr("/Users/colesmcintosh/Projects/open-xtract/test_docs/cm_resume.pdf"))