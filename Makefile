# ==========================================
# AI Obsidian Service - Enhanced Makefile
# Updated for stable dependencies and environment validation
# ==========================================

# Detect conda/mamba - prefer micromamba if available, then mamba, then conda
CONDA_EXE := $(shell which micromamba 2>/dev/null || which mamba 2>/dev/null || which conda 2>/dev/null)
ifeq ($(CONDA_EXE),)
    $(error No conda, mamba, or micromamba found in PATH)
endif

# Extract just the command name (micromamba, mamba, or conda)
CONDA := $(notdir $(CONDA_EXE))

CPU_ENV     := aiobs-cpu
GPU_ENV     := aiobs-gpu
PYVER       := 3.11

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
UVICORN_GPU   := $(PY_GPU) -m uvicorn

SRC := src
TESTS := tests

# Unit profile (aligns with CI unit job)
PYTEST_UNIT_EXPR := not integration_cpu and not integration_gpu and not e2e and not faiss
PYTEST_OPTS := -v --tb=short --strict-markers

# uvicorn timeouts
UVICORN_TIMEOUT  := 7200
UVICORN_GRACEFUL := 30

# ----------------------------
# Help
# ----------------------------
.PHONY: help
help:
	@echo "AI Obsidian Service - Development Commands"
	@echo ""
	@echo "Detected: $(CONDA) at $(CONDA_EXE)"
	@echo ""
	@echo "Environment Setup:"
	@echo "    env-create-cpu     Create CPU environment from environment.yml"
	@echo "    env-create-gpu     Create GPU environment from environment.gpu.yml"
	@echo "    env-validate-cpu   Quick validation of CPU environment"
	@echo "    env-validate-gpu   Quick validation of GPU environment"
	@echo "    env-remove-cpu     Remove CPU environment"
	@echo "    env-remove-gpu     Remove GPU environment"
	@echo ""
	@echo "Quality Checks:"
	@echo "    lint               Run ruff linting"
	@echo "    fmt                Auto-fix code formatting"
	@echo "    type               Run mypy type checking"
	@echo "    check              Full quality check (lint + type + test)"
	@echo ""
	@echo "Testing:"
	@echo "    test               Unit tests only (fast, no external deps)"
	@echo "    test-faiss-cpu     FAISS CPU integration tests"
	@echo "    test-faiss-gpu     FAISS GPU integration tests"
	@echo "    test-e2e           End-to-end tests"
	@echo "    test-all           Full test suite"
	@echo ""
	@echo "Development:"
	@echo "    serve-cpu          Run FastAPI server (CPU)"
	@echo "    serve-gpu          Run FastAPI server (GPU)"
	@echo "    verify-cpu         Comprehensive CPU diagnostics"
	@echo "    verify-gpu         Comprehensive GPU diagnostics"
	@echo "    clean              Clean build caches"

# ----------------------------
# Environment Management
# ----------------------------

.PHONY: env-create-cpu
env-create-cpu:
	@echo "Creating CPU environment from environment.yml..."
	-$(CONDA) env remove -n $(CPU_ENV) -y >/dev/null 2>&1 || true
	$(CONDA) env create -f environment.yml
	@echo "Installing project in development mode..."
	$(PIP_CPU) install -e ".[dev]"
	@echo "Validating environment..."
	@$(MAKE) env-validate-cpu
	@echo "CPU environment ready! Activate with: $(CONDA) activate $(CPU_ENV)"

.PHONY: env-create-gpu
env-create-gpu:
	@echo "Creating GPU environment from environment.gpu.yml..."
	-$(CONDA) env remove -n $(GPU_ENV) -y >/dev/null 2>&1 || true
	$(CONDA) env create -f environment.gpu.yml
	@echo "Installing project in development mode..."
	$(PIP_GPU) install -e ".[dev]"
	@echo "Validating environment..."
	@$(MAKE) env-validate-gpu
	@echo "GPU environment ready! Activate with: $(CONDA) activate $(GPU_ENV)"

.PHONY: env-validate-cpu
env-validate-cpu:
	@echo ">> Quick CPU environment validation ($(CPU_ENV))"
	@$(CONDA) run -n $(CPU_ENV) python -c "import torch, sentence_transformers, faiss; \
		print('Python OK | PyTorch OK | SentenceTransformers OK | FAISS OK')"
	@echo "Basic validation passed"

.PHONY: env-validate-gpu
env-validate-gpu:
	@echo ">> Quick GPU environment validation ($(GPU_ENV))"
	@$(CONDA) run -n $(GPU_ENV) python -c "import torch; \
		assert torch.cuda.is_available(), 'CUDA not available'"
	@$(CONDA) run -n $(GPU_ENV) python -c "import sentence_transformers, faiss; \
		print('Python OK | PyTorch OK | CUDA OK | SentenceTransformers OK | FAISS OK')"
	@echo "Basic validation passed"

.PHONY: env-remove-cpu
env-remove-cpu:
	@echo "Removing CPU environment..."
	@if [ "$$CONDA_DEFAULT_ENV" = "$(CPU_ENV)" ]; then \
		echo "Cannot remove active environment. Please run '$(CONDA) deactivate' first."; \
		exit 1; \
	fi
	-$(CONDA) env remove -n $(CPU_ENV) -y

.PHONY: env-remove-gpu
env-remove-gpu:
	@echo "Removing GPU environment..."
	@if [ "$$CONDA_DEFAULT_ENV" = "$(GPU_ENV)" ]; then \
		echo "Cannot remove active environment. Please run '$(CONDA) deactivate' first."; \
		exit 1; \
	fi
	-$(CONDA) env remove -n $(GPU_ENV) -y

# ----------------------------
# Quality Checks
# ----------------------------

.PHONY: lint
lint:
	@echo "Running ruff linting..."
	$(RUFF_CPU) check $(SRC) $(TESTS)

.PHONY: fmt
fmt:
	@echo "Auto-fixing code formatting..."
	$(RUFF_CPU) check $(SRC) $(TESTS) --fix

.PHONY: type
type:
	@echo "Running mypy type checking..."
	$(MYPY_CPU) $(SRC) $(TESTS)

.PHONY: check
check: lint type test
	@echo "All quality checks passed!"

# ----------------------------
# Tests
# ----------------------------

.PHONY: test
test:
	@echo "Running unit tests (memory backend, no external deps)..."
	VECTOR_STORE_BACKEND=memory AIOBS_TEST_MODE=1 $(PYTEST_CPU) $(PYTEST_OPTS) -m "$(PYTEST_UNIT_EXPR)"

.PHONY: test-faiss-cpu
test-faiss-cpu:
	@echo "Running FAISS CPU integration tests..."
	@$(MAKE) env-validate-cpu
	@echo "Testing FAISS functionality before running tests..."
	$(PY_CPU) -c "import faiss; print('FAISS', faiss.__version__, 'is working')"
	VECTOR_STORE_BACKEND=faiss $(PYTEST_CPU) $(PYTEST_OPTS) -m "integration_cpu or faiss"

.PHONY: test-faiss-gpu
test-faiss-gpu:
	@echo "Running FAISS GPU integration tests..."
	@$(MAKE) env-validate-gpu
	@echo "Testing FAISS GPU functionality..."
	$(PY_GPU) -c "import faiss; print('FAISS GPU', faiss.__version__, 'is working')"
	VECTOR_STORE_BACKEND=faiss $(PYTEST_GPU) $(PYTEST_OPTS) -m "integration_gpu"

.PHONY: test-e2e
test-e2e:
	@echo "Running end-to-end tests..."
	VECTOR_STORE_BACKEND=memory $(PYTEST_CPU) $(PYTEST_OPTS) -m "e2e"

.PHONY: test-all
test-all:
	@echo "Running full test suite..."
	$(PYTEST_CPU) $(PYTEST_OPTS)

# ----------------------------
# Development Servers
# ----------------------------

.PHONY: serve-cpu
serve-cpu:
	@if [ "$$CONDA_DEFAULT_ENV" != "$(CPU_ENV)" ]; then \
		echo "Error: Please activate the CPU environment first:"; \
		echo "  $(CONDA) activate $(CPU_ENV)"; \
		exit 1; \
	fi
	@echo "Starting FastAPI development server (CPU)..."
	uvicorn ai_obsidian_service.api.app:app --reload \
		--timeout-keep-alive $(UVICORN_TIMEOUT) \
		--timeout-graceful-shutdown $(UVICORN_GRACEFUL)

.PHONY: serve-gpu
serve-gpu:
	@if [ "$$CONDA_DEFAULT_ENV" != "$(GPU_ENV)" ]; then \
		echo "Error: Please activate the GPU environment first:"; \
		echo "  $(CONDA) activate $(GPU_ENV)"; \
		exit 1; \
	fi
	@echo "Starting FastAPI development server (GPU)..."
	uvicorn ai_obsidian_service.api.app:app --reload \
		--timeout-keep-alive $(UVICORN_TIMEOUT) \
		--timeout-graceful-shutdown $(UVICORN_GRACEFUL)

# ----------------------------
# Comprehensive Diagnostics
# ----------------------------

.PHONY: verify-cpu
verify-cpu:
	@echo "Comprehensive CPU environment verification..."
	@echo ""
	@echo "Package Versions:"
	@$(PY_CPU) -c "import sys; print('Python:', sys.version.split()[0])"
	@$(PY_CPU) -c "import numpy as np; print('NumPy:', np.__version__)"
	@$(PY_CPU) -c "import torch; print('PyTorch:', torch.__version__)"
	@$(PY_CPU) -c "import faiss; print('FAISS:', faiss.__version__)"
	@$(PY_CPU) -c "import sentence_transformers; print('SentenceTransformers:', sentence_transformers.__version__)"
	@echo ""
	@echo "PyTorch Info:"
	@$(PY_CPU) -c "import torch; print('CUDA available:', torch.cuda.is_available())"

.PHONY: verify-gpu
verify-gpu:
	@echo "Comprehensive GPU environment verification..."
	@echo ""
	@echo "Package Versions:"
	@$(PY_GPU) -c "import sys; print('Python:', sys.version.split()[0])"
	@$(PY_GPU) -c "import torch; print('PyTorch:', torch.__version__)"
	@$(PY_GPU) -c "import faiss; print('FAISS GPU:', faiss.__version__)"
	@$(PY_GPU) -c "import sentence_transformers; print('SentenceTransformers:', sentence_transformers.__version__)"
	@echo ""
	@echo "CUDA Status:"
	@$(PY_GPU) -c "import torch; print('CUDA available:', torch.cuda.is_available())"
	@$(PY_GPU) -c "import torch; print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
	@$(PY_GPU) -c "import torch; print('cuDNN version:', torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A')"
	@$(PY_GPU) -c "import torch; print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
	@$(PY_GPU) -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"
	@echo ""
	@echo "GPU Memory:"
	@$(PY_GPU) -c "import torch; props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; print(f'Total memory: {props.total_memory / 1024**3:.2f} GB') if props else print('GPU not available')"
	@$(PY_GPU) -c "import torch; props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; print(f'Available memory: {(props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB') if props else None"

.PHONY: clean
clean:
	@echo "Cleaning build caches..."
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "Caches cleaned"

# ----------------------------
# Setup Shortcuts
# ----------------------------

.PHONY: setup-cpu
setup-cpu: env-create-cpu
	@echo "CPU environment setup complete"

.PHONY: setup-gpu
setup-gpu: env-create-gpu
	@echo "GPU environment setup complete"
