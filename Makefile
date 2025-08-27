.PHONY: help install lint format typecheck test precommit precommit-install clean

help:
	@echo "Common targets:"
	@echo "  install            Install project with dev and vision extras"
	@echo "  lint               Run ruff checks"
	@echo "  format             Run ruff format and black"
	@echo "  typecheck          Run mypy on package"
	@echo "  test               Run pytest"
	@echo "  precommit          Run pre-commit on all files"
	@echo "  precommit-install  Install pre-commit git hook"

install:
	uv pip install -e .[dev,vision]

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run black .

typecheck:
	uv run mypy open_xtract

test:
	uv run pytest -q

precommit:
	uv run pre-commit run --all-files --show-diff-on-failure

precommit-install:
	uv run pre-commit install

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist **/*.egg-info
