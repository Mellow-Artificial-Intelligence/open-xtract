"""Offline benchmarks for openextract startup and local hotspots.

We don't (and can't) benchmark the LLM round-trip itself — that's network +
inference and dwarfs everything else. What we *can* measure is the local CPU
work that happens around it on every call: import cost, ``_get_media``, agent
construction, and extraction dispatch. Anything we shave here compounds across
``extract_many``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

# Stub a credential so pydantic-ai can construct an OpenAI client locally. This
# is not a real credential and is never used: no benchmark performs network I/O.
os.environ.setdefault("OPENAI_API_KEY", "sk-bench-dummy")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")

_COLD_RUNS = 5
_PROVIDER_DISTRIBUTIONS = (
    "openai",
    "anthropic",
    "google-genai",
    "botocore",
    "cohere",
    "huggingface-hub",
    "groq",
    "mistralai",
    "grpcio",
)
_COLD_IMPORT_CHILD = """
import json
import sys
import time

try:
    import resource
except ImportError:
    resource = None

start = time.perf_counter()
import openextract
elapsed = time.perf_counter() - start
if resource is None:
    max_rss_bytes = None
else:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_bytes = max_rss if sys.platform == "darwin" else max_rss * 1024
print(json.dumps({"elapsed": elapsed, "max_rss_bytes": max_rss_bytes}))
"""
_MODEL_ERROR_CHILD = """
import json
import sys
import time

try:
    import resource
except ImportError:
    resource = None

from pydantic_ai.exceptions import ModelAPIError
from openextract._extract import _map_exception

provider_prefixes = (
    "openai",
    "anthropic",
    "google.genai",
    "botocore",
    "cohere",
    "huggingface_hub",
    "groq",
    "mistralai",
    "grpc",
)
modules_before = set(sys.modules)
rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
start = time.perf_counter()
mapped = _map_exception(ModelAPIError(model_name="openai:gpt-5", message="benchmark failure"))
elapsed = time.perf_counter() - start
rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
rss_scale = 1 if sys.platform == "darwin" else 1024
new_provider_modules = sorted(
    module_name
    for module_name in set(sys.modules) - modules_before
    if any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in provider_prefixes
    )
)
print(
    json.dumps(
        {
            "elapsed": elapsed,
            "rss_delta_bytes": (
                max(0, rss_after - rss_before) * rss_scale
                if rss_before is not None and rss_after is not None
                else None
            ),
            "mapped_type": type(mapped).__name__,
            "new_provider_modules": new_provider_modules,
        }
    )
)
"""


class _Person(BaseModel):
    name: str
    age: int


def _fmt(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds * 1000:8.2f} ms"
    if seconds >= 1e-3:
        return f"{seconds * 1000:8.3f} ms"
    return f"{seconds * 1e6:8.2f} us"


def _fmt_bytes(size: float) -> str:
    return f"{size / (1024 * 1024):.2f} MiB"


def _run_child(code: str) -> dict:
    child_env = os.environ.copy()
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=child_env,
        text=True,
    )
    return json.loads(completed.stdout)


def bench_environment() -> None:
    from importlib import metadata

    installed_providers = []
    for distribution in _PROVIDER_DISTRIBUTIONS:
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
        installed_providers.append(f"{distribution}=={version}")

    if len(installed_providers) == len(_PROVIDER_DISTRIBUTIONS):
        profile = "full-provider development environment"
    elif installed_providers:
        profile = "partial-provider environment"
    else:
        profile = "base environment"

    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print("[environment]")
    print(f"  profile={profile}")
    print(f"  revision={revision or 'unknown'}")
    print(f"  python={platform.python_implementation()} {platform.python_version()}")
    print(f"  platform={platform.platform()}")
    print(
        "  providers="
        + (", ".join(installed_providers) if installed_providers else "none installed")
    )


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
    print("\n[import] cold `import openextract` in fresh subprocesses")
    samples = [_run_child(_COLD_IMPORT_CHILD) for _ in range(_COLD_RUNS)]
    timings = [sample["elapsed"] for sample in samples]
    max_rss = [sample["max_rss_bytes"] for sample in samples if sample["max_rss_bytes"]]
    timings.sort()
    print(
        f"  latency: median={_fmt(statistics.median(timings))}  "
        f"best={_fmt(timings[0])}  worst={_fmt(timings[-1])}"
    )
    if max_rss:
        print(
            f"  max RSS: median={_fmt_bytes(statistics.median(max_rss))}  "
            f"best={_fmt_bytes(min(max_rss))}  worst={_fmt_bytes(max(max_rss))}"
        )
    else:
        print("  max RSS: unavailable on this platform")


def bench_model_error_classification() -> None:
    print("\n[model error] first provider-neutral error classification")
    samples = [_run_child(_MODEL_ERROR_CHILD) for _ in range(_COLD_RUNS)]
    timings = sorted(sample["elapsed"] for sample in samples)
    rss_deltas = [
        sample["rss_delta_bytes"] for sample in samples if sample["rss_delta_bytes"] is not None
    ]
    mapped_types = {sample["mapped_type"] for sample in samples}
    new_provider_modules = sorted(
        {module_name for sample in samples for module_name in sample["new_provider_modules"]}
    )
    if mapped_types != {"ModelError"}:
        raise RuntimeError(f"Unexpected mapped exception types: {sorted(mapped_types)}")
    print(
        f"  latency: median={_fmt(statistics.median(timings))}  "
        f"best={_fmt(timings[0])}  worst={_fmt(timings[-1])}"
    )
    if rss_deltas:
        print(f"  max RSS delta: median={_fmt_bytes(statistics.median(rss_deltas))}")
    else:
        print("  max RSS delta: unavailable on this platform")
    print(
        "  newly imported provider modules: "
        + (", ".join(new_provider_modules) if new_provider_modules else "none")
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


def bench_build_agent() -> None:
    from openextract._extract import _build_agent

    print("\n[_build_agent] constructing a pydantic_ai.Agent")
    _bench(
        "_build_agent(non-ollama)",
        lambda: _build_agent(_Person, "openai:gpt-5", "extract"),
        iters=200,
    )
    _bench(
        "_build_agent(ollama)",
        lambda: _build_agent(_Person, "ollama:llama3", "extract"),
        iters=200,
    )


def bench_extract_end_to_end(tmp: Path) -> None:
    """End-to-end extract() with the LLM call mocked out — measures everything
    *except* the network/inference: media read, agent build, dispatch.
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
            "extract(path, openai:gpt-5)",
            lambda: extract(_Person, "openai:gpt-5", str(src)),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--startup-only",
        action="store_true",
        help="run only environment, cold-import, and first-error benchmarks",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bench_environment()
    bench_import_cost()
    bench_model_error_classification()

    if args.startup_only:
        return 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bench_get_media(tmp)
        bench_build_agent()
        bench_extract_end_to_end(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
