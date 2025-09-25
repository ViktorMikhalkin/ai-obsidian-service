# ==========================================
# AI Obsidian Service — Makefile (conda-based)
# Environments (fixed by convention):
#   - CPU env: aiobs-cpu
#   - GPU env: aiobs-gpu
#
# Requires: conda (or mamba/micromamba as a drop-in for "conda")
# Python: 3.12 (adjust if needed)
# ==========================================

CONDA       ?= conda
CPU_ENV     := aiobs-cpu
GPU_ENV     := aiobs-gpu
PYVER       := 3.12

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
	@echo "Targets:"
	@echo "  Environments:"
	@echo "    env-create-cpu     Create CPU env ($(CPU_ENV)) with Python $(PYVER), project, faiss-cpu"
	@echo "    env-create-gpu     Create GPU env ($(GPU_ENV)) with Python $(PYVER), project, faiss-gpu (+ CUDA req.)"
	@echo "    env-remove-cpu     Remove CPU env ($(CPU_ENV))"
	@echo "    env-remove-gpu     Remove GPU env ($(GPU_ENV))"
	@echo ""
	@echo "  Quality:"
	@echo "    lint               Run ruff"
	@echo "    fmt                Ruff --fix"
	@echo "    type               Run mypy"
	@echo "    check              lint + type + unit tests"
	@echo ""
	@echo "  Tests:"
	@echo "    test               Unit tests only (no faiss/integration/e2e)"
	@echo "    test-faiss-cpu     FAISS CPU tests (markers: integration_cpu or faiss)"
	@echo "    test-faiss-gpu     FAISS GPU tests (marker: integration_gpu)"
	@echo "    test-e2e           End-to-end tests (marker: e2e)"
	@echo "    test-all           Full suite (no marker filter)"
	@echo ""
	@echo "  Run:"
	@echo "    serve              Run FastAPI dev server (CPU env)"
	@echo ""
	@echo "  Verify:"
	@echo "    verify-cpu         Check Python/FAISS/Torch/ST (CPU)"
	@echo "    verify-gpu         Check Python/FAISS/Torch/ST + CUDA device (GPU)"
	@echo ""
	@echo "  Housekeeping:"
	@echo "    clean              Drop caches"

# ----------------------------
# Environments
# ----------------------------

.PHONY: env-create-cpu
env-create-cpu:
	@echo ">> Recreating CPU env $(CPU_ENV) with Python $(PYVER)"
	-$(CONDA) env remove -n $(CPU_ENV) -y >/dev/null 2>&1 || true
	$(CONDA) create -y -n $(CPU_ENV) python=$(PYVER) pip
	@echo ">> Installing project (editable) + dev extras"
	$(PIP_CPU) install --upgrade pip
	$(PIP_CPU) install -e ".[dev]"
	@echo ">> Installing FAISS CPU + sentence-transformers"
	-$(CONDA) install -y -n $(CPU_ENV) -c conda-forge faiss-cpu || true
	$(PIP_CPU) install sentence-transformers
	@echo ">> (Optional) Torch CPU:"
	@echo "   $(PIP_CPU) install torch --index-url https://download.pytorch.org/whl/cpu"

.PHONY: env-create-gpu
env-create-gpu:
	@echo ">> Recreating GPU env $(GPU_ENV) with Python $(PYVER)"
	-$(CONDA) env remove -n $(GPU_ENV) -y >/dev/null 2>&1 || true
	$(CONDA) create -y -n $(GPU_ENV) python=$(PYVER) pip
	@echo ">> Installing project (editable) + dev extras"
	$(PIP_GPU) install --upgrade pip
	$(PIP_GPU) install -e ".[dev]"
	@echo ">> Installing FAISS GPU + sentence-transformers"
	-$(CONDA) install -y -n $(GPU_ENV) -c conda-forge faiss-gpu || true
	$(PIP_GPU) install sentence-transformers
	@echo ">> Installing Torch (CUDA) — adjust index if needed"
	$(PIP_GPU) install torch --index-url https://download.pytorch.org/whl/cu121
	@echo ">> NOTE: Run GPU targets on a CUDA-capable machine/runner."

.PHONY: env-remove-cpu
env-remove-cpu:
	-$(CONDA) env remove -n $(CPU_ENV) -y

.PHONY: env-remove-gpu
env-remove-gpu:
	-$(CONDA) env remove -n $(GPU_ENV) -y

# ----------------------------
# Quality
# ----------------------------

.PHONY: lint
lint:
	$(RUFF_CPU) check $(SRC) $(TESTS)

.PHONY: fmt
fmt:
	$(RUFF_CPU) check $(SRC) $(TESTS) --fix

.PHONY: type
type:
	$(MYPY_CPU) $(SRC) $(TESTS)

.PHONY: check
check: lint type test
	@echo ">> All checks passed."

# ----------------------------
# Tests
# ----------------------------

.PHONY: test
test:
	@echo ">> Unit tests (CPU env, no faiss/integration/e2e)"
	VECTOR_STORE_BACKEND=memory AIOBS_TEST_MODE=1 $(PYTEST_CPU) $(PYTEST_OPTS) -m "$(PYTEST_UNIT_EXPR)"

.PHONY: test-faiss-cpu
test-faiss-cpu:
	@echo ">> FAISS CPU tests"
	VECTOR_STORE_BACKEND=faiss $(PYTEST_CPU) $(PYTEST_OPTS) -m "integration_cpu or faiss"

.PHONY: test-faiss-gpu
test-faiss-gpu:
	@echo ">> FAISS GPU tests"
	VECTOR_STORE_BACKEND=faiss $(PYTEST_GPU) $(PYTEST_OPTS) -m "integration_gpu"

.PHONY: test-e2e
test-e2e:
	@echo ">> E2E tests (CPU env)"
	VECTOR_STORE_BACKEND=memory $(PYTEST_CPU) $(PYTEST_OPTS) -m "e2e"

.PHONY: test-all
test-all:
	@echo ">> FULL SUITE (CPU env)"
	$(PYTEST_CPU) $(PYTEST_OPTS)

# ----------------------------
# Run
# ----------------------------

.PHONY: serve
serve:
	$(UVICORN_CPU) ai_obsidian_service.api.app:app --reload

# ----------------------------
# Verify environments (no heredocs — robust for Make)
# ----------------------------

.PHONY: verify-cpu
verify-cpu:
	@echo ">> Verifying CPU environment ($(CPU_ENV))"
	@printf '%s\n' \
	"import sys" \
	"print('Python:', sys.version)" \
	"try:" \
	"    import faiss" \
	"    print('FAISS available:', getattr(faiss, '__version__', 'unknown'))" \
	"except Exception as e:" \
	"    print('FAISS import FAILED:', e)" \
	"try:" \
	"    import torch" \
	"    print('Torch available:', torch.__version__)" \
	"    print('CUDA available:', torch.cuda.is_available())" \
	"except Exception as e:" \
	"    print('Torch import FAILED:', e)" \
	"try:" \
	"    import sentence_transformers" \
	"    print('SentenceTransformers:', sentence_transformers.__version__)" \
	"except Exception as e:" \
	"    print('SentenceTransformers import FAILED:', e)" \
	| $(PY_CPU) -

.PHONY: verify-gpu
verify-gpu:
	@echo ">> Verifying GPU environment ($(GPU_ENV))"
	@printf '%s\n' \
	"import sys" \
	"print('Python:', sys.version)" \
	"try:" \
	"    import faiss" \
	"    print('FAISS available:', getattr(faiss, '__version__', 'unknown'))" \
	"except Exception as e:" \
	"    print('FAISS import FAILED:', e)" \
	"try:" \
	"    import torch" \
	"    print('Torch available:', torch.__version__)" \
	"    print('CUDA available:', torch.cuda.is_available())" \
	"    " \
	"    # If CUDA is available, show device name" \
	"    if torch.cuda.is_available():" \
	"        print('CUDA device:', torch.cuda.get_device_name(0))" \
	"except Exception as e:" \
	"    print('Torch import FAILED:', e)" \
	"try:" \
	"    import sentence_transformers" \
	"    print('SentenceTransformers:', sentence_transformers.__version__)" \
	"except Exception as e:" \
	"    print('SentenceTransformers import FAILED:', e)" \
	| $(PY_GPU) -

# ----------------------------
# Housekeeping
# ----------------------------

.PHONY: clean
clean:
	@echo ">> Cleaning caches..."
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
