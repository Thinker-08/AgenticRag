PY        := .venv/bin/python
PIP       := .venv/bin/pip
AGRAG     := .venv/bin/agrag
UVICORN   := .venv/bin/uvicorn

DOC       ?= data/sample.txt
Q         ?= What is this document about?
TENANT    ?= default
CONFIG    ?= config/default.yaml
HOST      ?= 0.0.0.0
PORT      ?= 8000

.DEFAULT_GOAL := help

.PHONY: help venv install install-full pull-models up down run ingest ask eval lint fmt clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the ./.venv virtualenv
	python3 -m venv .venv
	$(PIP) install -U pip

install: venv ## Install the package + dev tools (local mode, no GPU/services)
	$(PIP) install -e '.[dev]'

install-full: venv ## Install every extra (ml/pdf/stores/obs/eval/dev) for the full stack
	$(PIP) install -e '.[ml,pdf,stores,obs,eval,dev]'

pull-models: ## Pull the Gemma tags into the running Ollama (gemma3:12b + gemma3:4b)
	docker compose exec ollama ollama pull gemma3:12b
	docker compose exec ollama ollama pull gemma3:4b

up: ## Start the full stack (Ollama, Qdrant, Redis, Langfuse, app)
	docker compose up -d

down: ## Stop the stack (add ARGS=-v to also drop volumes)
	docker compose down $(ARGS)

run: ## Run the API locally with uvicorn (AGRAG_CONFIG=$(CONFIG))
	AGRAG_CONFIG=$(CONFIG) $(UVICORN) agrag.serving.app:app --host $(HOST) --port $(PORT) --reload

ingest: ## Ingest a document: make ingest DOC=path/to/file
	AGRAG_CONFIG=$(CONFIG) $(AGRAG) ingest --tenant $(TENANT) $(DOC)

ask: ## Ask a question: make ask Q="your question"
	AGRAG_CONFIG=$(CONFIG) $(AGRAG) ask --tenant $(TENANT) "$(Q)"

eval: ## Run the golden-set eval harness
	AGRAG_CONFIG=$(CONFIG) $(AGRAG) eval

lint: ## Lint with ruff
	$(PY) -m ruff check src

fmt: ## Auto-format + fix with ruff
	$(PY) -m ruff format src
	$(PY) -m ruff check --fix src

clean: ## Remove caches and build artifacts (keeps .venv and data/)
	rm -rf build dist src/*.egg-info .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
