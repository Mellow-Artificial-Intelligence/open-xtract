"""Command-line interface for openextract."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from typing import Any, cast

from pydantic import BaseModel

from ._extract import extract, extract_many, extract_with_usage
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


def _resolve_input_files(
    raw_inputs: list[str],
    *,
    media_type: str | None,
) -> list[str | bytes]:
    """Resolve CLI paths, URLs, or stdin ``-`` to values accepted by ``extract``."""
    if "-" in raw_inputs:
        if len(raw_inputs) > 1:
            raise ValueError("stdin (-) cannot be combined with other input files")
        if not media_type:
            raise ValueError("--media-type is required when reading from stdin (-)")
        return [sys.stdin.buffer.read()]

    return cast(list[str | bytes], raw_inputs)


def _usage_payload(usage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _print_json(payload: Any, *, as_repr: bool) -> None:
    if as_repr:
        print(repr(payload))
        return
    print(json.dumps(payload, indent=2, default=str))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openextract",
        description="Extract structured data from files or URLs using an LLM.",
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        metavar="input_file",
        help="One or more paths/URLs, or '-' to read bytes from stdin.",
    )
    parser.add_argument(
        "--schema",
        required=True,
        help="Pydantic model import path in 'module:ClassName' form.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="pydantic-ai model identifier (e.g. 'xai:grok-4.3').",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Optional natural-language instructions for the model.",
    )
    parser.add_argument(
        "--media-type",
        default=None,
        metavar="MIME",
        help="MIME type (required for stdin; optional override for paths/URLs).",
    )
    parser.add_argument(
        "--usage",
        action="store_true",
        help="Print token usage (single input only).",
    )
    parser.add_argument(
        "--output",
        choices=("json", "repr"),
        default="json",
        help="Output format: 'json' (default) or 'repr'.",
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
        input_files = _resolve_input_files(args.input_files, media_type=args.media_type)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.usage and len(input_files) != 1:
        print("error: --usage requires exactly one input file", file=sys.stderr)
        return 1

    media_type = args.media_type

    try:
        if len(input_files) == 1:
            input_file = input_files[0]
            if args.usage:
                result, usage = extract_with_usage(
                    schema=schema_cls,
                    model=args.model,
                    input_file=input_file,
                    instructions=args.instructions,
                    media_type=media_type,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                )
                payload: Any = {"result": result.model_dump(), "usage": _usage_payload(usage)}
            else:
                payload = extract(
                    schema=schema_cls,
                    model=args.model,
                    input_file=input_file,
                    instructions=args.instructions,
                    media_type=media_type,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                )
        else:
            results = extract_many(
                schema=schema_cls,
                model=args.model,
                input_files=input_files,
                instructions=args.instructions,
                media_type=media_type,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
            )
            payload = [r.model_dump() for r in results]
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

    if args.output == "repr":
        _print_json(payload, as_repr=True)
    elif len(input_files) == 1 and not args.usage and isinstance(payload, BaseModel):
        print(payload.model_dump_json(indent=2))
    else:
        _print_json(payload, as_repr=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
