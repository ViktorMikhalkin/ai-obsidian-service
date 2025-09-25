# ==========================================
# AI Obsidian Service – Enhanced Makefile
# Updated for stable dependencies and environment validation
# ==========================================

CONDA       ?= conda
CPU_ENV     := aiobs-cpu
GPU_ENV     := aiobs-gpu
PYVER       := 3.11  # Updated to stable version

# Tools run INSIDE the envs
PY_CPU        := $(CONDA) run -n $(CPU_ENV) python
PIP_CPU       := $(PY_CPU) -m pip
PYTEST_CPU    := $(PY_CPU) -m pytest
RUFF_CPU      := $(PY_CPU) -m ruff
MYPY_CPU      := $(PY_CPU) -m mypy
UVICORN_CPU   := $(PY_CPU) -m uvicorn

PY_GPU        := $(CONDA) run -n $(GPU_ENV) python
PIP_GPU       := $(PY_GPU) -m pip
PYTEST_GPU    := $(PY_GPU) -m pytest

SRC := src
TESTS := tests

# Unit profile (aligns with CI unit job)
PYTEST_UNIT_EXPR := not integration_cpu and not integration_gpu and not e2e and not faiss
PYTEST_OPTS := -v --tb=short --strict-markers

# ----------------------------
# Help
# ----------------------------
.PHONY: help
help:
	@echo "AI Obsidian Service - Development Commands"
	@echo ""
	@echo "🏗️  Environment Setup:"
	@echo "    env-create-cpu     Create CPU environment from environment.yml"
	@echo "    env-create-gpu     Create GPU environment from environment.yml"
	@echo "    env-validate-cpu   Validate CPU environment compatibility"
	@echo "    env-validate-gpu   Validate GPU environment compatibility"
	@echo "    env-remove-cpu     Remove CPU environment"
	@echo "    env-remove-gpu     Remove GPU environment"
	@echo ""
	@echo "✅ Quality Checks:"
	@echo "    lint               Run ruff linting"
	@echo "    fmt                Auto-fix code formatting"
	@echo "    type               Run mypy type checking"
	@echo "    check              Full quality check (lint + type + env-validate + tests)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "    test               Unit tests only (fast, no external deps)"
	@echo "    test-faiss-cpu     FAISS CPU integration tests"
	@echo "    test-faiss-gpu     FAISS GPU integration tests"
	@echo "    test-e2e           End-to-end tests"
	@echo "    test-all           Full test suite"
	@echo ""
	@echo "🚀 Development:"
	@echo "    serve              Run FastAPI development server"
	@echo "    verify-cpu         Verify CPU environment setup"
	@echo "    verify-gpu         Verify GPU environment setup"
	@echo "    clean              Clean build caches"

# ----------------------------
# Environment Management
# ----------------------------

.PHONY: env-create-cpu
env-create-cpu:
	@echo "🏗️  Creating CPU environment from environment.yml..."
	-$(CONDA) env remove -n $(CPU_ENV) -y >/dev/null 2>&1 || true
	$(CONDA) env create -f environment.yml
	@echo "📦 Installing project in development mode..."
	$(PIP_CPU) install -e ".[dev]"
	@echo "✅ Validating environment..."
	@$(MAKE) env-validate-cpu
	@echo "🎉 CPU environment ready! Activate with: conda activate $(CPU_ENV)"

.PHONY: env-create-gpu
env-create-gpu:
	@echo "🏗️  Creating GPU environment from environment.gpu.yml..."
	-$(CONDA) env remove -n $(GPU_ENV) -y >/dev/null 2>&1 || true
	$(CONDA) env create -f environment.gpu.yml
	@echo "📦 Installing project in development mode..."
	$(PIP_GPU) install -e ".[dev]"
	@echo "✅ Validating environment..."
	@CONDA_ENV=$(GPU_ENV) $(MAKE) env-validate-gpu
	@echo "🎉 GPU environment ready! Activate with: conda activate $(GPU_ENV)"

.PHONY: env-validate-cpu
env-validate-cpu:
	@echo ">> Validating CPU environment ($(CPU_ENV))"
	$(CONDA) run -n $(CPU_ENV) python -c "import sys; print('Python', sys.version)"
	$(CONDA) run -n $(CPU_ENV) python -c "import torch; print('Torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
	$(CONDA) run -n $(CPU_ENV) python -c "import sentence_transformers; print('SentenceTransformers', sentence_transformers.__version__)"
	-$(CONDA) run -n $(CPU_ENV) python -c "import faiss; print('FAISS', faiss.__version__)" || echo "FAISS not installed"

.PHONY: env-validate-gpu
env-validate-gpu:
	@echo ">> Validating GPU environment ($(GPU_ENV))"
	$(CONDA) run -n $(GPU_ENV) python -c "import sys; print('Python', sys.version)"
	$(CONDA) run -n $(GPU_ENV) python -c "import torch; print('Torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); \
		print('Device count:', torch.cuda.device_count()); \
		print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
	$(CONDA) run -n $(GPU_ENV) python -c "import sentence_transformers; print('SentenceTransformers', sentence_transformers.__version__)"
	-$(CONDA) run -n $(GPU_ENV) python -c "import faiss; print('FAISS', faiss.__version__)" || echo "FAISS not installed"

.PHONY: env-remove-cpu
env-remove-cpu:
	@echo "Removing CPU environment..."
	@if [ "$CONDA_DEFAULT_ENV" = "$(CPU_ENV)" ]; then \
		echo "Cannot remove active environment. Please run 'conda deactivate' first."; \
		exit 1; \
	fi
	-$(CONDA) env remove -n $(CPU_ENV) -y

.PHONY: env-remove-gpu
env-remove-gpu:
	@echo "Removing GPU environment..."
	@if [ "$CONDA_DEFAULT_ENV" = "$(GPU_ENV)" ]; then \
		echo "Cannot remove active environment. Please run 'conda deactivate' first."; \
		exit 1; \
	fi
	-$(CONDA) env remove -n $(GPU_ENV) -y

# ----------------------------
# Quality (Enhanced with validation)
# ----------------------------

.PHONY: lint
lint:
	@echo "🔍 Running ruff linting..."
	$(RUFF_CPU) check $(SRC) $(TESTS)

.PHONY: fmt
fmt:
	@echo "🎨 Auto-fixing code formatting..."
	$(RUFF_CPU) check $(SRC) $(TESTS) --fix

.PHONY: type
type:
	@echo "🔍 Running mypy type checking..."
	$(MYPY_CPU) $(SRC) $(TESTS)

.PHONY: check
check: lint type test
	@echo "✅ All quality checks passed!"

# ----------------------------
# Tests
# ----------------------------

.PHONY: test
test:
	@echo "🧪 Running unit tests (memory backend, no external deps)..."
	VECTOR_STORE_BACKEND=memory AIOBS_TEST_MODE=1 $(PYTEST_CPU) $(PYTEST_OPTS) -m "$(PYTEST_UNIT_EXPR)"

.PHONY: test-faiss-cpu
test-faiss-cpu:
	@echo "🧪 Running FAISS CPU integration tests..."
	@$(MAKE) env-validate-cpu
	@echo "🔧 Testing FAISS functionality before running tests..."
	$(PY_CPU) -c "import faiss; print('✅ FAISS', faiss.__version__, 'is working')"
	VECTOR_STORE_BACKEND=faiss $(PYTEST_CPU) $(PYTEST_OPTS) -m "integration_cpu or faiss"

.PHONY: test-faiss-gpu
test-faiss-gpu:
	@echo "🧪 Running FAISS GPU integration tests..."
	@$(MAKE) env-validate-gpu
	@echo "🔧 Testing FAISS GPU functionality..."
	$(PY_GPU) -c "import faiss; print('✅ FAISS GPU', faiss.__version__, 'is working')"
	VECTOR_STORE_BACKEND=faiss $(PYTEST_GPU) $(PYTEST_OPTS) -m "integration_gpu"

.PHONY: test-e2e
test-e2e:
	@echo "🧪 Running end-to-end tests..."
	VECTOR_STORE_BACKEND=memory $(PYTEST_CPU) $(PYTEST_OPTS) -m "e2e"

.PHONY: test-all
test-all:
	@echo "🧪 Running full test suite..."
	$(PYTEST_CPU) $(PYTEST_OPTS)

# ----------------------------
# Development
# ----------------------------

.PHONY: serve
serve:
	@echo "🚀 Starting FastAPI development server..."
	$(UVICORN_CPU) ai_obsidian_service.api.app:app --reload

# ----------------------------
# Environment Verification (Enhanced)
# ----------------------------

.PHONY: verify-cpu
verify-cpu:
	@echo "🔍 Comprehensive CPU environment verification..."
	$(PY_CPU) -m ai_obsidian_service.utils.environment
	@echo ""
	@echo "📋 Package Versions:"
	$(PY_CPU) -c "import sys; print('Python:', sys.version.split()[0])"
	$(PY_CPU) -c "import numpy as np; print('NumPy:', np.__version__)"
	$(PY_CPU) -c "import torch; print('PyTorch:', torch.__version__)"
	-$(PY_CPU) -c "import faiss; print('FAISS:', faiss.__version__)" || echo "FAISS: not available"
	-$(PY_CPU) -c "import sentence_transformers; print('SentenceTransformers:', sentence_transformers.__version__)" || echo "SentenceTransformers: not available"

.PHONY: verify-gpu
verify-gpu:
	@echo "🔍 Comprehensive GPU environment verification..."
	$(PY_GPU) -m ai_obsidian_service.utils.environment
	@echo ""
	@echo "📋 Package Versions:"
	$(PY_GPU) -c "import sys; print('Python:', sys.version.split()[0])"
	$(PY_GPU) -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
	-$(PY_GPU) -c "import faiss; print('FAISS GPU:', faiss.__version__)" || echo "FAISS GPU: not available"
	-$(PY_GPU) -c "import sentence_transformers; print('SentenceTransformers:', sentence_transformers.__version__)" || echo "SentenceTransformers: not available"

# ----------------------------
# Housekeeping
# ----------------------------

.PHONY: clean
clean:
	@echo "🧹 Cleaning build caches..."
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Caches cleaned"

# ----------------------------
# Setup Commands
# ----------------------------

.PHONY: setup-cpu
setup-cpu:
	@echo "Setting up CPU environment..."
	@$(MAKE) env-create-cpu

.PHONY: setup-gpu
setup-gpu:
	@echo "Setting up GPU environment..."
	@$(MAKE) env-create-gpu