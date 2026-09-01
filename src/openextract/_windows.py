"""Run one extraction per parse window and merge the results."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from ._citations import split_cited_output
from ._parse import ParsedDocument, parsed_window_inputs
from ._reduce import reduce_outputs
from ._retry import _run_with_retries_async, _run_with_retries_sync
from ._types import Citation, T, Usage, _sum_usage

WindowRun = Callable[[list], tuple[object, Usage]]
AsyncWindowRun = Callable[[list], Awaitable[tuple[object, Usage]]]


def extract_windows_sync(
    run: WindowRun,
    inputs: list,
    parsed: ParsedDocument | None,
    schema: type[T],
    cite: bool,
    *,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
) -> tuple[T, Usage, tuple[Citation, ...]]:
    """Extract each parse window (or the one-window fast path) and merge."""
    windows = parsed_window_inputs(parsed, inputs)
    outputs: list[T] = []
    usages: list[Usage] = []
    citations: list[Citation] = []
    for window in windows:

        def _once(
            window: list = window,
        ) -> tuple[T, Usage, tuple[Citation, ...]]:
            raw, usage = run(window)
            output, cites = split_cited_output(raw, schema, cite=cite, parsed=parsed)
            return output, usage, cites

        output, usage, cites = _run_with_retries_sync(
            _once,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        )
        outputs.append(output)
        usages.append(usage)
        citations.extend(cites)
    return _fold_windows(outputs, usages, citations)


async def extract_windows_async(
    run: AsyncWindowRun,
    inputs: list,
    parsed: ParsedDocument | None,
    schema: type[T],
    cite: bool,
    *,
    max_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
) -> tuple[T, Usage, tuple[Citation, ...]]:
    """Async sibling of :func:`extract_windows_sync`."""
    windows = parsed_window_inputs(parsed, inputs)
    outputs: list[T] = []
    usages: list[Usage] = []
    citations: list[Citation] = []
    for window in windows:

        async def _once(
            window: list = window,
        ) -> tuple[T, Usage, tuple[Citation, ...]]:
            raw, usage = await run(window)
            output, cites = split_cited_output(raw, schema, cite=cite, parsed=parsed)
            return output, usage, cites

        output, usage, cites = await _run_with_retries_async(
            _once,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_max_backoff=retry_max_backoff,
        )
        outputs.append(output)
        usages.append(usage)
        citations.extend(cites)
    return _fold_windows(outputs, usages, citations)


def _fold_windows(
    outputs: Sequence[T],
    usages: Sequence[Usage],
    citations: list[Citation],
) -> tuple[T, Usage, tuple[Citation, ...]]:
    """Keep the single-window object identity; merge only when chunked."""
    if len(outputs) == 1:
        return outputs[0], usages[0], tuple(citations)
    return reduce_outputs(outputs), _sum_usage(usages), tuple(citations)
