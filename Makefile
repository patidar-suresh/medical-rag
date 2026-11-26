.PHONY: install lint format typecheck test check up down doctor

install:        ## Create .venv (Python 3.11) and install the project with dev tools
	uv venv --python 3.11
	uv pip install -e ".[dev]"
	uv run pre-commit install

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

check: lint typecheck test

up:             ## Start Qdrant (Ollama runs separately on the host)
	docker compose up -d qdrant

down:
	docker compose down

doctor:
	uv run medrag doctor
