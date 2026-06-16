# Orkhon developer Makefile.
#
# Targets:
#   setup  - sync the environment with uv
#   test   - run the unit/integration suite (excludes slow end-to-end runs)
#   smoke  - run the full pipeline on the tiny smoke configs
#   fmt    - format the codebase (black + ruff if available)
#   clean  - remove caches and generated run/export artifacts

PY := .venv/bin/python
UV_CACHE_DIR ?= .uv-cache

.DEFAULT_GOAL := help
.PHONY: help setup test smoke fmt clean

help:
	@echo "Orkhon make targets:"
	@echo "  setup  - uv sync (install/update dependencies)"
	@echo "  test   - pytest -m 'not slow'"
	@echo "  smoke  - bash scripts/smoke_all.sh (full pipeline)"
	@echo "  fmt    - format with black + ruff"
	@echo "  clean  - remove caches and generated artifacts"

setup:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync

test:
	$(PY) -m pytest -m 'not slow'

smoke:
	bash scripts/smoke_all.sh

fmt:
	-$(PY) -m black src tests
	-$(PY) -m ruff check --fix src tests

clean:
	rm -rf .pytest_cache .ruff_cache
	rm -rf runs exports artifacts data/prepared
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

lint:
	uv run ruff check src/ tests/ || true
	uv run python -m py_compile src/orkhon/cli.py
