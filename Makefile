# ==== AI-Obsidian Makefile (fixed) ====
# Usage:
#   make env-gpu         # create GPU env (Python 3.12 + CUDA 12.1 PyTorch)
#   make env-cpu         # create CPU env (Python 3.12)
#   make install         # pip install project requirements in env (incl. FastAPI)
#   make check-cuda      # print CUDA status in the chosen env
#   make build           # run index build
#   make status          # show number of chunks in index
#   make serve           # run API server with autoreload
#   make serve-no-reload # run API server without autoreload
#   make clean           # remove index/
#
# Override variables:
#   make ENV=aiobs-gpu build
#   make PY=3.12 ENV=aiobs-gpu env-gpu

ENV ?= aiobs-cpu
PY  ?= 3.12
CONDA_RUN := conda run -n $(ENV)

# Channels are set per-env to avoid global config drift
CHANNELS = -c pytorch -c nvidia

.PHONY: help env-gpu env-cpu install install-test-deps setup-cpu setup-gpu check-cuda build status serve serve-no-reload clean test test-v build-test

help:
	@echo "Targets:"
	@echo "  env-cpu            Create CPU conda env (Python $(PY))"
	@echo "  env-gpu            Create GPU conda env (Python $(PY), CUDA 12.1 torch)"
	@echo "  install            Pip install requirements.txt into $(ENV)"
	@echo "  install-test-deps  Install test-only deps (pytest) into $(ENV)"
	@echo "  setup-cpu          env-cpu + install + install-test-deps"
	@echo "  setup-gpu          env-gpu + install + install-test-deps"
	@echo "  check-cuda         Print CUDA diagnostics (uses $(ENV))"
	@echo "  build              Build index (uses $(ENV))"
	@echo "  status             Show index stats (uses $(ENV))"
	@echo "  serve              Run API with reload (uses $(ENV))"
	@echo "  serve-no-reload    Run API without reload (uses $(ENV))"
	@echo "  test               Run pytest in safe mode (AIOBS_TEST_MODE=1)"
	@echo "  test-v             Same, verbose"
	@echo "  build-test         Build index in safe mode (AIOBS_TEST_MODE=1)"
	@echo "  clean              Remove index/"

# ---- Environments ----
env-gpu:
	conda env remove -n $(ENV) -y || true
	conda create -n $(ENV) -y python=$(PY)
	conda run -n $(ENV) conda config --env --add channels pytorch
	conda run -n $(ENV) conda config --env --add channels nvidia
	conda run -n $(ENV) conda config --env --set channel_priority strict
	conda run -n $(ENV) conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 $(CHANNELS)
	$(MAKE) ENV=$(ENV) install

env-cpu:
	conda env remove -n $(ENV) -y || true
	conda create -n $(ENV) -y python=$(PY)
	conda run -n $(ENV) pip install --upgrade pip
	$(MAKE) ENV=$(ENV) install

# Convenience setups
setup-cpu:
	$(MAKE) ENV=$(ENV) env-cpu
	$(MAKE) ENV=$(ENV) install-test-deps

setup-gpu:
	$(MAKE) ENV=$(ENV) env-gpu
	$(MAKE) ENV=$(ENV) install-test-deps

# ---- Project deps (pip) ----
install:
	$(CONDA_RUN) pip install -U -r requirements.txt

install-test-deps:
	$(CONDA_RUN) pip install -U pytest

# ---- Diagnostics ----
check-cuda:
	@echo ">> Checking CUDA in env $(ENV)"
	$(CONDA_RUN) python -c "import torch, os; print('torch:', torch.__version__); print('built_with_cuda:', torch.version.cuda); print('cuda_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES')); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '<none>')"

# ---- App tasks ----
build:
	$(CONDA_RUN) python -m cli.aiobs build

status:
	$(CONDA_RUN) python -m cli.aiobs status

serve:
	$(CONDA_RUN) python -m cli.aiobs serve --host 127.0.0.1 --port 8000

serve-no-reload:
	$(CONDA_RUN) python - <<'PY'\
import uvicorn; uvicorn.run("indexer.app:app", host="127.0.0.1", port=8000, reload=False)\
PY

clean:
	rm -rf index

# ---- Tests / Safe builds ----
test:
	$(CONDA_RUN) env AIOBS_TEST_MODE=1 python -m pytest -q

test-v:
	$(CONDA_RUN) env AIOBS_TEST_MODE=1 python -m pytest -v

# Build in safe mode (does not touch real index_dir). Optionally set AIOBS_TEST_INDEX_DIR=/tmp/aiobs-test
build-test:
	$(CONDA_RUN) env AIOBS_TEST_MODE=1 python -m cli.aiobs build
