# Makefile for AI ↔ Obsidian — Local Indexing & RAG Service (conda-first)

SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c

ENV    ?= aiobs-cpu
PY     ?= 3.12
CONDA  ?= $(shell command -v mamba >/dev/null 2>&1 && echo mamba || echo conda)

.PHONY: env-cpu env-gpu setup-cpu setup-gpu install reinstall install-test-deps \
	  pre-commit help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup-gpu       - Create GPU environment and install dependencies"
	@echo "  setup-cpu       - Create CPU environment (fallback)"
	@echo "  build           - Build the index (GPU-accelerated by default)"
	@echo "  serve           - Start the API service"
	@echo "  status          - Show index status"
	@echo ""
	@echo "Development:"
	@echo "  test            - Run all tests"
	@echo "  test-coverage   - Run tests with coverage report"
	@echo "  test-all-envs   - Test both GPU and CPU environments"
	@echo "  lint            - Check code with ruff"
	@echo "  lint-fix        - Auto-fix ruff issues"
	@echo "  format          - Format code with ruff"
	@echo "  typecheck       - Run mypy type checking"
	@echo "  quality         - Run all quality checks (lint + typecheck + test)"
	@echo "  quality-all-envs - Quality checks on both environments"
	@echo "  pre-commit      - Run pre-commit checks (format-check + lint + typecheck)"
	@echo ""
	@echo "Environment Setup:"
	@echo "  dev-gpu         - Setup GPU development environment"
	@echo "  dev-cpu         - Setup CPU development environment"
	@echo "  dev-both        - Setup both development environments"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean           - Remove local index"
	@echo "  clean-env       - Remove conda environment"
	@echo "  check-gpu       - Verify GPU setup"

# --- CPU-only environment (fast setup) ---
env-cpu:
	@echo "Creating CPU-only environment: $(ENV)"
	@if [ ! -f environment.yml ]; then \
		echo "Error: environment.yml not found!"; \
		exit 1; \
	fi
	$(CONDA) env create -n $(ENV) -f environment.yml || \
	$(CONDA) env update -n $(ENV) -f environment.yml
	@echo "Setting strict channel priority..."
	$(CONDA) run -n $(ENV) conda config --env --set channel_priority strict

# --- GPU environment (recommended for CUDA 12.4) ---
env-gpu:
	@echo "Creating GPU environment: $(ENV) from environment.gpu.yml"
	@if [ ! -f environment.gpu.yml ]; then \
		echo "Error: environment.gpu.yml not found!"; \
		exit 1; \
	fi
	$(CONDA) env create -n $(ENV) -f environment.gpu.yml || \
	$(CONDA) env update -n $(ENV) -f environment.gpu.yml
	@echo "Setting strict channel priority..."
	$(CONDA) run -n $(ENV) conda config --env --set channel_priority strict

# --- Complete setup with test dependencies ---
setup-cpu: env-cpu
	@echo "CPU environment $(ENV) is ready!"

setup-gpu: env-gpu
	@echo "GPU environment $(ENV) is ready!"
	@echo "Verifying GPU setup..."
	@$(MAKE) check-cuda
	@$(MAKE) check-faiss

install:
	@echo "Dependencies managed via conda environment - nothing to install"

reinstall:
	@echo "Dependencies managed via conda environment - use 'make clean-env && make setup-gpu' to refresh"

install-test-deps:
	@echo "Test dependencies managed via conda environment - nothing to install"

# --- Index building ---
build:
	@echo "Building index..."
	@if [ "$(ENV)" = "aiobs-cpu" ] || [[ "$(ENV)" == *-cpu ]]; then \
		echo "Using CPU-only mode"; \
		mkdir -p scripts; \
		echo '#!/bin/bash' > scripts/run_cpu.sh; \
		echo 'export CUDA_VISIBLE_DEVICES=""' >> scripts/run_cpu.sh; \
		echo 'export TORCH_DEVICE=cpu' >> scripts/run_cpu.sh; \
		echo 'export PYTHONPATH=$(PWD)/src' >> scripts/run_cpu.sh; \
		echo 'python -m ai_obsidian_service.cli.aiobs "$@"' >> scripts/run_cpu.sh; \
		chmod +x scripts/run_cpu.sh; \
		$(CONDA) run -n $(ENV) bash scripts/run_cpu.sh build; \
	else \
		$(CONDA) run -n $(ENV) env PYTHONPATH=$(PWD)/src python -m ai_obsidian_service.cli.aiobs build; \
	fi

build-test:
	@echo "Building index in test mode..."
	$(CONDA) run -n $(ENV) env PYTHONPATH=$(PWD)/src AIOBS_TEST_MODE=1 python -m ai_obsidian_service.cli.aiobs build

# --- Index status ---
status:
	@echo "Checking index status..."
	$(CONDA) run -n $(ENV) env PYTHONPATH=$(PWD)/src python -c "from ai_obsidian_service.indexer.status import main as _m; import sys; sys.exit(_m())"

serve:
	@echo "Starting API service at http://0.0.0.0:8000"
	@if [ "$(ENV)" = "aiobs-cpu" ] || [[ "$(ENV)" == *-cpu ]]; then \
		echo "Using CPU-only mode"; \
		$(CONDA) run -n $(ENV) env PYTHONPATH=$(PWD)/src CUDA_VISIBLE_DEVICES="" TORCH_DEVICE=cpu uvicorn ai_obsidian_service.indexer.app:app --reload --host 0.0.0.0 --port 8000; \
	else \
		$(CONDA) run -n $(ENV) env PYTHONPATH=$(PWD)/src uvicorn ai_obsidian_service.indexer.app:app --reload --host 0.0.0.0 --port 8000; \
	fi

# --- Testing ---
test:
	@echo "Running tests..."
	$(CONDA) run -n $(ENV) pytest -v

test-coverage:
	@echo "Running tests with coverage..."
	$(CONDA) run -n $(ENV) pytest --cov=ai_obsidian_service --cov-report=html --cov-report=term

test-integration:
	@echo "Running integration tests..."
	$(CONDA) run -n $(ENV) pytest tests/integration/ -v

# --- Code Quality ---
lint:
	@echo "Running ruff linter..."
	$(CONDA) run -n $(ENV) ruff check src/ tests/

lint-fix:
	@echo "Auto-fixing ruff issues..."
	$(CONDA) run -n $(ENV) ruff check --fix src/ tests/

format:
	@echo "Formatting code with ruff..."
	$(CONDA) run -n $(ENV) ruff format src/ tests/

format-check:
	@echo "Checking code formatting..."
	$(CONDA) run -n $(ENV) ruff format --check src/ tests/

typecheck:
	@echo "Running mypy type checker..."
	$(CONDA) run -n $(ENV) mypy src/

# --- Cross-environment testing ---
test-all-envs:
	@echo "Testing both GPU and CPU environments..."
	@if conda env list | grep -q aiobs-gpu; then \
		echo "Testing GPU environment..."; \
		ENV=aiobs-gpu $(MAKE) test; \
	else \
		echo "GPU environment not found, skipping..."; \
	fi
	@if conda env list | grep -q aiobs-cpu; then \
		echo "Testing CPU environment..."; \
		ENV=aiobs-cpu $(MAKE) test; \
	else \
		echo "CPU environment not found, skipping..."; \
	fi

quality-all-envs:
	@echo "Quality checks on both environments..."
	@if conda env list | grep -q aiobs-gpu; then \
		echo "Quality checks on GPU environment..."; \
		ENV=aiobs-gpu $(MAKE) quality; \
	else \
		echo "GPU environment not found, skipping..."; \
	fi
	@if conda env list | grep -q aiobs-cpu; then \
		echo "Quality checks on CPU environment..."; \
		ENV=aiobs-cpu $(MAKE) quality; \
	else \
		echo "CPU environment not found, skipping..."; \
	fi

# --- Environment-specific development ---
dev-gpu:
	@echo "Setting up GPU development environment..."
	@$(MAKE) setup-gpu ENV=aiobs-gpu
	@$(MAKE) check-gpu ENV=aiobs-gpu
	@echo "Ready for GPU development! Use: ENV=aiobs-gpu make <command>"

dev-cpu:
	@echo "Setting up CPU development environment..."
	@$(MAKE) setup-cpu ENV=aiobs-cpu
	@echo "Ready for CPU development! Use: ENV=aiobs-cpu make <command>"

dev-both:
	@echo "Setting up both development environments..."
	@$(MAKE) dev-gpu
	@$(MAKE) dev-cpu
	@echo "Both environments ready!"

# --- Check CUDA/PyTorch setup ---
check-cuda:
	@echo "Checking CUDA/PyTorch setup..."
	@$(CONDA) run -n $(ENV) python -c "import torch; \
print('torch:', torch.__version__); \
print('cuda_available:', torch.cuda.is_available()); \
print('device_count:', torch.cuda.device_count()); \
print('cuda_version:', getattr(__import__('torch').version, 'cuda', None)); \
print('device_name[0]:', __import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'N/A')" || \
	(echo "❌ CUDA check failed!"; exit 1)
	@echo "✅ CUDA setup OK!"

# --- Check faiss-gpu setup ---
check-faiss:
	@echo "Checking faiss-gpu setup..."
	@$(CONDA) run -n $(ENV) python -c "import faiss; \
print('faiss-gpu available:', hasattr(faiss, 'StandardGpuResources')); \
res = faiss.StandardGpuResources() if hasattr(faiss, 'StandardGpuResources') else None; \
print('GPU resources initialized successfully' if res else 'GPU resources not available')" || \
	(echo "❌ faiss-gpu check failed!"; exit 1)
	@echo "✅ faiss-gpu setup OK!"

# --- Complete GPU environment verification ---
check-gpu: check-cuda check-faiss
	@echo "✅ All GPU components verified!"

# --- Remove local index ---
clean:
	@echo "Cleaning local index..."
	rm -rf .index/

# --- Clean environment ---
clean-env:
	@echo "Removing conda environment: $(ENV)"
	$(CONDA) env remove -n $(ENV) -y || true

# --- Environment management from YAML/lock ---
env-from-yml:
	@if [ "$(ENV)" = "aiobs-gpu" ] || [[ "$(ENV)" == *-gpu ]]; then \
		echo "Creating GPU environment from YAML..."; \
		$(CONDA) env create -f environment.gpu.yml -n $(ENV) || \
		$(CONDA) env update -f environment.gpu.yml -n $(ENV); \
	else \
		echo "Creating CPU environment from YAML..."; \
		$(CONDA) env create -f environment.yml -n $(ENV) || \
		$(CONDA) env update -f environment.yml -n $(ENV); \
	fi
	$(CONDA) run -n $(ENV) conda config --env --set channel_priority strict

env-from-lock:
	@echo "Installing from conda-lock..."
	conda-lock install --name $(ENV)

# --- Environment information ---
info:
	@echo "Environment: $(ENV)"
	@echo "Python version: $(PY)"
	@echo "Conda command: $(CONDA)"
	@$(CONDA) run -n $(ENV) conda list | head -20 || echo "Environment $(ENV) not found"

# --- Quick start options ---
quickstart-cpu:
	@echo "🚀 Quick start with CPU environment..."
	@$(MAKE) setup-cpu
	@$(MAKE) build-test
	@echo "✅ Ready! Run 'make serve' to start the service"

quickstart-gpu:
	@echo "🚀 Quick start with GPU environment..."
	@$(MAKE) setup-gpu
	@$(MAKE) build-test
	@echo "✅ Ready! Run 'make serve' to start the service"
