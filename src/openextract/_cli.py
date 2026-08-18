"""Command-line interface for openextract."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from typing import Any, BinaryIO, cast

from dotenv import load_dotenv
from pydantic import BaseModel

from ._agents import DefinedAgent, RemoteAgent, load_agent, load_agents
from ._batch import extract_many
from ._extract import extract, extract_with_usage
from ._reduce import SwarmReduce
from ._styles import ExtractionStyle
from ._swarm import extract_swarm, extract_swarm_with_results
from .exceptions import (
    ExtractionError,
    ModelError,
    ProviderNotInstalledError,
    RemoteAgentError,
    SchemaValidationError,
    UrlFetchError,
)


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


def _split_list(value: str | None) -> list[str]:
    """Split a comma-separated CLI list, dropping blank entries."""
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _resolve_agents(args) -> list[DefinedAgent | RemoteAgent]:
    """Load the agents named by ``--agent`` / ``--agents``."""
    if args.agent and args.agents:
        raise ValueError("use --agent or --agents, not both")
    if args.agent:
        return [load_agent(args.agent)]
    if args.agents:
        return load_agents(args.agents)
    return []


def _agent_schema(agents: list[DefinedAgent | RemoteAgent]) -> type[BaseModel] | None:
    """Return the first output schema declared by the loaded agents, if any."""
    for agent in agents:
        if agent.output_schema is not None:
            return agent.output_schema
    return None


def _swarm_agents(args, agents: list[DefinedAgent | RemoteAgent]) -> list | None:
    """Build the swarm agent list, or ``None`` when this is a one-shot run.

    ``--agents`` and ``--models`` list the agents outright; ``--swarm`` fans a
    single model out. A lone ``--agent`` is not a swarm — it may still fan out
    on its own if it declares subagents, which ``extract`` handles.
    """
    models = _split_list(args.models)
    if len(agents) > 1:
        return agents
    if len(models) > 1:
        return models
    if args.swarm > 1:
        return agents or models or [args.model]
    return None


def _validate_swarm_args(args, agents: list, inputs: list) -> None:
    """Reject swarm flag combinations before any input is loaded."""
    if args.swarm < 1:
        raise ValueError("--swarm must be a positive integer")
    models = _split_list(args.models)
    if (args.swarm > 1 or len(models) > 1 or agents) and len(inputs) != 1:
        raise ValueError(
            "--swarm, --models, --agent, and --agents apply to a single input; "
            "omit them for batch files"
        )
    if len(models) > 1 and args.swarm > 1 and args.swarm != len(models):
        raise ValueError("--swarm does not match the number of --models")
    if args.models and args.model:
        raise ValueError("use --model or --models, not both")


def _resolve_input_files(
    raw_inputs: list[str],
    *,
    media_type: str | None,
) -> list[str | bytes | BinaryIO]:
    """Resolve CLI paths, URLs, or stdin ``-`` to values accepted by ``extract``."""
    if "-" in raw_inputs:
        if len(raw_inputs) > 1:
            raise ValueError("stdin (-) cannot be combined with other input files")
        if not media_type:
            raise ValueError("--media-type is required when reading from stdin (-)")
        return [sys.stdin.buffer]

    return cast(list[str | bytes | BinaryIO], raw_inputs)


def _usage_payload(usage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _batch_payload(
    results: list,
    inputs: list[str | bytes | BinaryIO],
) -> tuple[list[Any], int]:
    """Build the JSON entries for a batch run and count per-item failures.

    Successful items serialize to their model dump. Failed items (only present
    when ``--continue-on-error`` runs the batch with ``return_exceptions=True``)
    become an error object tagged with the originating input.
    """
    payload: list[Any] = []
    failures = 0
    for item, result in zip(inputs, results, strict=True):
        if isinstance(result, BaseException):
            failures += 1
            payload.append(
                {
                    "input": item if isinstance(item, str) else "<bytes>",
                    "error": str(result),
                    "error_type": type(result).__name__,
                }
            )
        else:
            payload.append(result.model_dump())
    return payload, failures


def _run_swarm(args, schema_cls, swarm_agents: list, input_file) -> Any:
    """Run a swarm over one input and build its CLI payload."""
    options = {
        "instructions": args.instructions,
        "size": args.swarm if len(swarm_agents) == 1 else None,
        "style": args.style,
        "reduce": args.reduce,
        "media_type": args.media_type,
        "max_input_bytes": args.max_input_bytes,
        "max_retries": args.max_retries,
        "retry_backoff": args.retry_backoff,
        "retry_max_backoff": args.retry_max_backoff,
    }
    if not args.usage:
        return extract_swarm(schema_cls, swarm_agents, input_file, **options)
    swarm = extract_swarm_with_results(schema_cls, swarm_agents, input_file, **options)
    return {
        "result": swarm.output.model_dump(),
        "usage": _usage_payload(swarm.usage),
        "agents": len(swarm.agents),
        "reduce": swarm.reduce.value,
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
        default=None,
        help=(
            "Pydantic model import path in 'module:ClassName' form. Optional "
            "when --agent/--agents supplies an output schema."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="pydantic-ai model identifier (e.g. 'xai:grok-4.3').",
    )
    parser.add_argument(
        "--models",
        default=None,
        metavar="ID,ID",
        help="Comma-separated model identifiers, one per swarm agent.",
    )
    parser.add_argument(
        "--agent",
        default=None,
        metavar="SPEC",
        help="Agent to extract with: a directory, a Python file, or 'module:attribute'.",
    )
    parser.add_argument(
        "--agents",
        default=None,
        metavar="SPEC,SPEC",
        help="Comma-separated agent specs, one per swarm agent.",
    )
    parser.add_argument(
        "--swarm",
        type=int,
        default=1,
        metavar="N",
        help="Run N parallel agents over a single input (default 1).",
    )
    parser.add_argument(
        "--reduce",
        choices=tuple(item.value for item in SwarmReduce),
        default=SwarmReduce.MERGE.value,
        help="How a swarm folds agent outputs: 'merge' (default), 'vote', or 'first'.",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Optional natural-language instructions for the model.",
    )
    parser.add_argument(
        "--style",
        choices=tuple(item.value for item in ExtractionStyle),
        default=ExtractionStyle.DIRECT.value,
        help=(
            "Extraction style: 'direct' (default) sends media to the model; "
            "'search' uses file tools on text; 'code' writes Python against text."
        ),
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
        "--continue-on-error",
        action="store_true",
        help=(
            "Batch only: keep going when an input fails; report per-item errors "
            "inline and exit 7 if any failed (default: abort on first failure)."
        ),
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
        "--max-input-bytes",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum bytes to load per input (default: OPENEXTRACT_MAX_INPUT_BYTES or 52428800)."
        ),
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Base backoff in seconds for retry (default 1.0).",
    )
    parser.add_argument(
        "--retry-max-backoff",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Maximum retry delay in seconds, including Retry-After (default 60.0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the openextract CLI. Returns the exit code."""
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        agents = _resolve_agents(args)
        schema_cls = _resolve_schema(args.schema) if args.schema else _agent_schema(agents)
        if schema_cls is None:
            raise ValueError(
                "--schema is required unless --agent/--agents supplies an output schema"
            )
        if not args.model and not args.models and not agents:
            raise ValueError("--model, --models, or --agent/--agents is required")
        input_files = _resolve_input_files(args.input_files, media_type=args.media_type)
        _validate_swarm_args(args, agents, input_files)
    except (ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.usage and len(input_files) != 1:
        print("error: --usage requires exactly one input file", file=sys.stderr)
        return 1

    swarm_agents = _swarm_agents(args, agents)
    single_model = agents[0] if agents else (args.model or _split_list(args.models)[0])
    media_type = args.media_type
    batch_failures = 0

    try:
        if swarm_agents is not None:
            payload = _run_swarm(args, schema_cls, swarm_agents, input_files[0])
        elif len(input_files) == 1:
            input_file = input_files[0]
            if args.usage:
                result, usage = extract_with_usage(
                    schema=schema_cls,
                    model=single_model,
                    input_file=input_file,
                    instructions=args.instructions,
                    style=args.style,
                    media_type=media_type,
                    max_input_bytes=args.max_input_bytes,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                    retry_max_backoff=args.retry_max_backoff,
                )
                payload: Any = {"result": result.model_dump(), "usage": _usage_payload(usage)}
            else:
                payload = extract(
                    schema=schema_cls,
                    model=single_model,
                    input_file=input_file,
                    instructions=args.instructions,
                    style=args.style,
                    media_type=media_type,
                    max_input_bytes=args.max_input_bytes,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                    retry_max_backoff=args.retry_max_backoff,
                )
        else:
            results = extract_many(
                schema=schema_cls,
                model=cast(str, single_model),
                input_files=input_files,
                instructions=args.instructions,
                style=args.style,
                media_type=media_type,
                max_input_bytes=args.max_input_bytes,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                retry_max_backoff=args.retry_max_backoff,
                return_exceptions=args.continue_on_error,
            )
            payload, batch_failures = _batch_payload(results, input_files)
    except UrlFetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SchemaValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except RemoteAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 8
    except ProviderNotInstalledError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 6
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output == "repr":
        _print_json(payload, as_repr=True)
    elif not args.usage and isinstance(payload, BaseModel):
        print(payload.model_dump_json(indent=2))
    else:
        _print_json(payload, as_repr=False)

    if batch_failures:
        print(
            f"warning: {batch_failures} of {len(input_files)} input(s) failed; "
            "see output for details",
            file=sys.stderr,
        )
        return 7  # partial batch failure under --continue-on-error
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
