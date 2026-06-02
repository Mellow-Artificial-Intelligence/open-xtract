"""Microbenchmarks for openextract local hotspots.

We don't (and can't) benchmark the LLM round-trip itself — that's network +
inference and dwarfs everything else. What we *can* measure is the local CPU
work that happens around it on every call: import cost, ``_get_media``,
agent construction, and the per-call ``load_dotenv`` overhead. Anything we
shave here compounds across ``extract_many``.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

# Stub credentials so pydantic-ai provider clients can be instantiated locally.
os.environ.setdefault("OPENAI_API_KEY", "sk-bench-dummy")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")


class _Person(BaseModel):
    name: str
    age: int


def _fmt(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds * 1000:8.2f} ms"
    if seconds >= 1e-3:
        return f"{seconds * 1000:8.3f} ms"
    return f"{seconds * 1e6:8.2f} us"


def _bench(label: str, fn, *, iters: int, warmup: int = 1) -> None:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    median = statistics.median(samples)
    p95 = samples[int(0.95 * (len(samples) - 1))]
    best = samples[0]
    print(f"  {label:<42} median={_fmt(median)}  p95={_fmt(p95)}  best={_fmt(best)}  (n={iters})")


def bench_import_cost() -> None:
    print("\n[import] cold-start cost of `import openextract` (subprocess)")
    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", "import openextract"],
            check=True,
            capture_output=True,
        )
        timings.append(time.perf_counter() - t0)
    timings.sort()
    print(
        f"  median={_fmt(statistics.median(timings))}  "
        f"best={_fmt(timings[0])}  worst={_fmt(timings[-1])}"
    )


def bench_get_media(tmp: Path) -> None:
    from openextract._extract import _get_media, _get_media_type

    small = tmp / "small.txt"
    small.write_bytes(b"hello world")
    medium = tmp / "medium.pdf"
    medium.write_bytes(b"%PDF" + b"x" * 50_000)
    large = tmp / "large.pdf"
    large.write_bytes(b"%PDF" + b"x" * 5_000_000)

    print("\n[_get_media] resolving input_file to (bytes, media_type)")
    _bench("path: small text (~11 B)", lambda: _get_media(str(small)), iters=2000)
    _bench("path: medium pdf (~50 KB)", lambda: _get_media(str(medium)), iters=1000)
    _bench("path: large pdf (~5 MB)", lambda: _get_media(str(large)), iters=200)

    raw = b"x" * 100_000
    _bench(
        "bytes input (100 KB, just type-check)",
        lambda: _get_media(raw, media_type="application/pdf"),
        iters=20000,
    )

    print("\n[_get_media_type] mimetypes.guess_type")
    _bench("guess_type('foo.pdf')", lambda: _get_media_type("foo.pdf"), iters=20000)


def bench_load_dotenv() -> None:
    from dotenv import load_dotenv

    print("\n[dotenv] per-call cost of load_dotenv()  (called inside every extract!)")
    _bench("load_dotenv() (no .env present)", lambda: load_dotenv(), iters=2000)


def bench_build_agent() -> None:
    from openextract._extract import _build_agent

    print("\n[_build_agent] constructing a pydantic_ai.Agent")
    _bench(
        "_build_agent(non-ollama)",
        lambda: _build_agent(_Person, "xai:grok-4.3", "extract"),
        iters=200,
    )
    _bench(
        "_build_agent(ollama)",
        lambda: _build_agent(_Person, "ollama:llama3", "extract"),
        iters=200,
    )


def bench_extract_end_to_end(tmp: Path) -> None:
    """End-to-end extract() with the LLM call mocked out — measures everything
    *except* the network/inference: dotenv, media read, agent build, dispatch.
    """
    from openextract import extract, extract_many

    src = tmp / "doc.pdf"
    src.write_bytes(b"%PDF" + b"x" * 50_000)

    agent_instance = MagicMock()
    run_result = MagicMock()
    run_result.output = _Person(name="Ada", age=36)
    agent_instance.run_sync.return_value = run_result

    print("\n[extract] sync extract() with Agent mocked (all-local cost per call)")
    with patch("openextract._extract.Agent", return_value=agent_instance):
        _bench(
            "extract(path, xai:grok-4.3)",
            lambda: extract(_Person, "xai:grok-4.3", str(src)),
            iters=500,
        )

    print("\n[extract_many] 20 inputs, max_concurrency=5, Agent mocked")
    files = []
    for i in range(20):
        p = tmp / f"many_{i}.pdf"
        p.write_bytes(b"%PDF" + b"x" * 20_000)
        files.append(str(p))

    async_agent_instance = MagicMock()
    from unittest.mock import AsyncMock

    async_agent_instance.run = AsyncMock(return_value=run_result)
    with patch("openextract._extract.Agent", return_value=async_agent_instance):
        _bench(
            "extract_many(20 files, conc=5)",
            lambda: extract_many(_Person, "xai:grok-4.3", files, max_concurrency=5),
            iters=20,
        )


def main() -> int:
    bench_import_cost()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bench_get_media(tmp)
        bench_load_dotenv()
        bench_build_agent()
        bench_extract_end_to_end(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
