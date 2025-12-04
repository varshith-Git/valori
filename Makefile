# Makefile for project automation: env, install, test, lint, format, typecheck, security scans
# Designed for macOS / zsh users; Make runs with /bin/sh but commands use portable patterns.

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PRECOMMIT := $(VENV)/bin/pre-commit

.PHONY: help create-env install downloads test coverage lint flake format black black-check format-check typecheck safety bandit \
	deep-analysis ci clean install-dev


help:
	@echo "Available targets:"
	@echo "  create-env    - create a virtualenv at $(VENV) and install dev requirements"
	@echo "  install       - install this package in editable mode into the venv"
	@echo "  downloads     - pre-download parsing/dependency wheels into ./deps"
	@echo "  test          - run pytest (uses pytest.ini / pyproject settings)"
	@echo "  coverage      - run pytest with coverage report"
	@echo "  lint          - run black-check, isort (check) and flake8"
	@echo "  flake         - run flake8 only"
	@echo "  format        - run black and isort to format code (alias for 'make black && make isort')"
	@echo "  black         - run black formatter on the repo (uses venv)"
	@echo "  isort         - run isort to sort imports (uses venv)"
	@echo "  black-check   - run black --check to verify formatting"
	@echo "  typecheck     - run mypy"
	@echo "  safety        - run safety check (requires safety in the venv)"
	@echo "  bandit        - run bandit security scan"
	@echo "  deep-analysis - run typecheck, flake8, bandit, safety and tests"
	@echo "  ci            - full pipeline: create-env, install, downloads, deep-analysis, coverage"
	@echo "  clean         - cleanup build artifacts and optionally remove $(VENV)"

# Create a virtualenv and install requirements
create-env:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	@if [ -f requirements-dev.txt ]; then \
		$(PIP) install -r requirements-dev.txt; \
	else \
		echo "requirements-dev.txt not found; please install dev tools manually"; \
	fi

install: ## install package into the virtualenv (editable)
	@if [ ! -f $(VENV)/bin/activate ]; then \
		echo "Virtualenv not found, creating $(VENV)"; \
		$(MAKE) create-env; \
	fi
	$(PIP) install -e .

install-dev: install
	@if [ -x "$(PRECOMMIT)" ]; then \
		$(PRECOMMIT) install || true; \
	else \
		echo "pre-commit not installed in venv; run 'make create-env' or 'make install'"; \
	fi

# downloads: pre-download parsing dependencies into deps/
downloads:
	mkdir -p deps
	@if [ -f requirements-parsing.txt ]; then \
		$(PIP) download -r requirements-parsing.txt -d deps || true; \
	else \
		echo "requirements-parsing.txt not found; skipping downloads"; \
	fi
	@if [ -x ./scripts/install_dev.sh ]; then \
		./scripts/install_dev.sh || true; \
	fi

test:
	@$(PY) -m pytest

coverage:
	@$(PY) -m pytest --cov=valori --cov-report=term-missing --cov-report=xml

lint: flake format-check

flake:
	@$(VENV)/bin/flake8 src tests || true

black-check:
	@$(VENV)/bin/black --check . || true

black:
	@$(VENV)/bin/black .

isort:
	@$(VENV)/bin/isort .

format-check: black-check
	@$(VENV)/bin/isort --check-only . || true

format: black
	@$(VENV)/bin/isort .

typecheck:
	@$(VENV)/bin/mypy src || true

safety:
	@$(VENV)/bin/safety check || true

bandit:
	@$(VENV)/bin/bandit -r src -ll || true

deep-analysis: typecheck flake bandit safety test

ci: create-env install downloads deep-analysis coverage

clean:
	@echo "Removing build artifacts and caches..."
	@rm -rf build dist *.egg-info
	@find . -type d -name __pycache__ -exec rm -rf {} + || true
	@find . -type f -name '*.pyc' -delete || true
	@echo "Artifacts removed. To remove the virtualenv run: rm -rf $(VENV)"
