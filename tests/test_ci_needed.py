"""Path-filter decisions for CI jobs."""

import runpy
from pathlib import Path

jobs_for_paths = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "ci_needed.py")
)["jobs_for_paths"]


def test_docs_only_skips_python_jobs():
    assert jobs_for_paths(["docs/cli.md", "README.md"]) == {
        "lint": False,
        "test": False,
        "package": False,
    }


def test_src_change_runs_all_jobs():
    assert jobs_for_paths(["src/openextract/_extract.py"]) == {
        "lint": True,
        "test": True,
        "package": True,
    }


def test_tests_skip_package_smoke():
    assert jobs_for_paths(["tests/test_cli.py"]) == {
        "lint": True,
        "test": True,
        "package": False,
    }


def test_api_reference_runs_lint_only():
    assert jobs_for_paths(["docs/api-reference.md"]) == {
        "lint": True,
        "test": False,
        "package": False,
    }


def test_lockfile_runs_all_jobs():
    assert jobs_for_paths(["uv.lock"]) == {
        "lint": True,
        "test": True,
        "package": True,
    }


def test_examples_run_lint_and_tests():
    assert jobs_for_paths(["examples/basic/local_file.py"]) == {
        "lint": True,
        "test": True,
        "package": False,
    }


def test_scripts_run_lint_only():
    assert jobs_for_paths(["scripts/bench.py"]) == {
        "lint": True,
        "test": False,
        "package": False,
    }


def test_empty_change_set_skips_all_jobs():
    assert jobs_for_paths([]) == {
        "lint": False,
        "test": False,
        "package": False,
    }


def test_ci_workflow_change_runs_all_jobs():
    assert jobs_for_paths([".github/workflows/ci.yml"]) == {
        "lint": True,
        "test": True,
        "package": True,
    }
