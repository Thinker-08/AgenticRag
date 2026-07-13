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

help:
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:
	python3 -m venv .venv
	$(PIP) install -U pip

install: venv
	$(PIP) install -e '.[dev]'

install-full: venv
	$(PIP) install -e '.[ml,pdf,stores,obs,eval,dev]'

pull-models:
	docker compose exec ollama ollama pull gemma3:12b
	docker compose exec ollama ollama pull gemma3:4b

up:
	docker compose up -d

down:
	docker compose down $(ARGS)

run:
	AGRAG_CONFIG=$(CONFIG) $(UVICORN) agrag.serving.app:app --host $(HOST) --port $(PORT) --reload

ingest:
	AGRAG_CONFIG=$(CONFIG) $(AGRAG) ingest --tenant $(TENANT) $(DOC)

ask:
	AGRAG_CONFIG=$(CONFIG) $(AGRAG) ask --tenant $(TENANT) "$(Q)"

eval:
	AGRAG_CONFIG=$(CONFIG) $(AGRAG) eval

lint:
	$(PY) -m ruff check src

fmt:
	$(PY) -m ruff format src
	$(PY) -m ruff check --fix src

clean:
	rm -rf build dist src/*.egg-info .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
