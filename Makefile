.PHONY: check_uv, check_python, install, install_dev, check, test, docs, docs-test, update, help

check_uv: # install `uv` if not installed
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "uv is not installed, installing now..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

check_python: check_uv
		@if ! which python3.12 > /dev/null 2>&1; then \
		echo "Python 3.12 not found, creating Conda environment..."; \
		conda create --name py312 python=3.12 --yes && \
		conda_path=$$(conda env list | grep 'py312' | awk '{print $$NF}') && \
		echo "Conda environment created: $$conda_path"; \
		export PATH="$$conda_path/bin:$$PATH"; \
		if [ -n "$$CONDA_PREFIX" ]; then \
			echo "An active Conda environment is detected: $$CONDA_PREFIX"; \
			conda deactivate; \
		fi; \
	fi; \
	uv venv --python=3.11 --seed

install: check_python
	@echo "📦 Installing dependencies"
	@. .venv/bin/activate && \
		uv pip compile pyproject.toml -o requirements.txt && \
		uv pip install -r requirements.txt && \
		uv pip install -e . && \
		pre-commit install

install_dev: check_python
	@echo "📦 Installing dependencies"
	@. .venv/bin/activate && \
		uv pip compile pyproject.toml --extra=dev -o requirements-dev.txt && \
		uv pip install -r requirements-dev.txt && \
		uv pip install -e . && \
		pre-commit install

check: ## Run code quality tools.
	@echo "🧹 Checking code: Running ruff and pre-commit"
	@. .venv/bin/activate && \
		echo "⚡️ Linting code: Running ruff" && \
		ruff check . && \
		echo "🧹 Checking code: Running pre-commit" && \
		pre-commit run --all-files

test: ## run tests
	@echo "✅ Testing code: Running pytest"
	@. .venv/bin/activate && pytest

update: ## Update pre-commit hooks
	@echo "⚙️ Updating environment and pre-commit hooks"
	@. .venv/bin/activate && \
		pre-commit autoupdate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help