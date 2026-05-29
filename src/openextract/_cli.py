"""Command-line interface for openextract."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from collections.abc import Sequence

from pydantic import BaseModel

from ._extract import extract
from .exceptions import ExtractionError, ModelError, SchemaValidationError, UrlFetchError


def _resolve_schema(schema_path: str) -> type[BaseModel]:
    """Resolve a ``module:ClassName`` string to a Pydantic ``BaseModel`` subclass."""
    if ":" not in schema_path:
        raise ValueError(
            f"Invalid schema path '{schema_path}'. Expected format 'module:ClassName'."
        )

    module_part, class_name = schema_path.split(":", 1)
    if not module_part or not class_name:
        raise ValueError(
            f"Invalid schema path '{schema_path}'. Expected format 'module:ClassName'."
        )

    module = importlib.import_module(module_part)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"Class '{class_name}' not found in module '{module_part}'.") from exc

    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise ValueError(f"'{schema_path}' does not refer to a Pydantic BaseModel subclass.")

    return cls


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openextract",
        description="Extract structured data from a file or URL using an LLM.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('openextract')}",
    )
    parser.add_argument(
        "input_file",
        help="Path or URL of the document, image, audio, or video to extract from.",
    )
    parser.add_argument(
        "--schema",
        required=True,
        help="Pydantic model import path in 'module:ClassName' form.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="pydantic-ai model identifier (e.g. 'openai:gpt-5').",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Optional natural-language instructions for the model.",
    )
    parser.add_argument(
        "--output",
        choices=("json", "repr"),
        default="json",
        help="Output format: 'json' (default) prints model_dump_json; 'repr' prints repr().",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        metavar="N",
        help="Retry up to N times on ModelError with exponential backoff (default 0).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Base backoff in seconds for retry (default 1.0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the openextract CLI. Returns the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        schema_cls = _resolve_schema(args.schema)
    except (ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        result = extract(
            schema=schema_cls,
            model=args.model,
            input_file=args.input_file,
            instructions=args.instructions,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )
    except UrlFetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SchemaValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5

    if args.output == "json":
        print(result.model_dump_json(indent=2))
    else:
        print(repr(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
