# Minimal Makefile for uv-based workflows

.PHONY: help setup run run-dev test test-api doctor clean

HOST ?= 0.0.0.0
PORT ?= 8000

help:
	@echo "Available targets:"
	@echo "  setup     - Create venv and install deps via uv sync"
	@echo "  run       - Start API server (uvicorn)"
	@echo "  run-dev   - Start API server with reload"
	@echo "  test      - Run pytest"
	@echo "  test-api  - Hit local endpoints with example payloads"
	@echo "  doctor    - Check environment and setup requirements"
	@echo "  clean     - Remove .venv and __pycache__"

setup:
	uv sync

run:
	uv run uvicorn api.main:app --host $(HOST) --port $(PORT)

run-dev:
	uv run uvicorn api.main:app --host $(HOST) --port $(PORT) --reload

test:
	uv run pytest -q tests/test_api.py

test-api:
	uv run python scripts/test_endpoints.py

doctor:
	@echo "==> Checking uv installation"
	@command -v uv >/dev/null 2>&1 && echo "✔ uv found" || { echo "✘ uv not found. Install: https://docs.astral.sh/uv/"; exit 1; }
	@echo "==> Checking virtual environment (managed by uv)"
	@test -d .venv && echo "✔ .venv present" || echo "ℹ No .venv yet (will be created on 'make setup')"
	@echo "==> Checking .env file"
	@test -f .env && echo "✔ .env present" || echo "ℹ No .env file found (optional if OPENROUTER_API_KEY exported)"
	@echo "==> Checking OPENROUTER_API_KEY availability"
	@if [ -n "$$OPENROUTER_API_KEY" ]; then \
		echo "✔ OPENROUTER_API_KEY found in environment"; \
	elif [ -f .env ] && grep -E '^[[:space:]]*OPENROUTER_API_KEY[[:space:]]*=' .env >/dev/null; then \
		echo "✔ OPENROUTER_API_KEY found in .env"; \
	else \
		echo "✘ OPENROUTER_API_KEY not found. Add to environment or .env"; exit 1; \
	fi
	@echo "==> Verifying required Python packages via uv"
	@uv run python -c "import fastapi, uvicorn; print('✔ fastapi/uvicorn import ok')" 2>/dev/null || { echo "✘ Missing deps. Run: make setup"; exit 1; }
	@echo "All checks passed. You're good to go."

clean:
	rm -rf .venv **/__pycache__

