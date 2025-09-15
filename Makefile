# ==== AI-Obsidian Makefile (conda-only) ====
# Usage:
#   make setup-cpu       # create/update CPU env from environment.yml
#   make setup-gpu       # create/update GPU env from environment.gpu.yml
#   make check-cuda      # print CUDA status in the chosen env
#   make build           # run index build
#   make status          # show number of chunks in index
#   make serve           # run API server with autoreload
#   make serve-no-reload # run API server without autoreload
#   make clean           # remove index/
#
# Override variables:
#   make ENV=aiobs-gpu build
#   make PY=3.12 ENV=aiobs-gpu setup-gpu

ENV ?= aiobs-cpu
PY  ?= 3.12
CONDA_RUN := conda run -n $(ENV)

.PHONY: help env-cpu env-gpu setup-cpu setup-gpu check-cuda build status serve serve-no-reload clean test test-v build-test

help:
	@echo "Targets:"
	@echo "  setup-cpu          Create/Update CPU env from environment.yml"
	@echo "  setup-gpu          Create/Update GPU env from environment.gpu.yml"
	@echo "  check-cuda         Print CUDA diagnostics (uses $(ENV))"
	@echo "  build              Build index (uses $(ENV))"
	@echo "  status             Show index stats (uses $(ENV))"
	@echo "  serve              Run API with reload (uses $(ENV))"
	@echo "  serve-no-reload    Run API without reload (uses $(ENV))"
	@echo "  test               Run pytest in safe mode (AIOBS_TEST_MODE=1)"
	@echo "  test-v             Same, verbose"
	@echo "  build-test         Build index in safe mode (AIOBS_TEST_MODE=1)"
	@echo "  clean              Remove index/"

# Create/Update CPU environment from environment.yml
setup-cpu:
	@conda env list | grep -E '^$(ENV)\s' >/dev/null || conda env create -n $(ENV) -f environment.yml
	@conda env update -n $(ENV) -f environment.yml --prune
	@conda install -n $(ENV) -c conda-forge pre-commit -y
	@conda run -n $(ENV) pre-commit install
	@conda run -n $(ENV) python -c "import sys; print('Python:', sys.version)"

# Create/Update GPU environment from environment.gpu.yml
setup-gpu:
	@conda env list | grep -E '^$(ENV)\s' >/dev/null || conda env create -n $(ENV) -f environment.gpu.yml
	@conda env update -n $(ENV) -f environment.gpu.yml --prune
	@conda install -n $(ENV) -c conda-forge pre-commit -y
	@conda run -n $(ENV) pre-commit install
	@conda run -n $(ENV) python -c "import sys; print('Python:', sys.version)"

check-cuda:
	@echo ">> Checking CUDA in env $(ENV)"
	$(CONDA_RUN) python -c "import os; \
try: import torch; v=torch.__version__; cuda=getattr(torch.version,'cuda',None); avail=torch.cuda.is_available(); cnt=torch.cuda.device_count(); name=(torch.cuda.get_device_name(0) if avail else '<none>'); \
except Exception as e: v,cuda,avail,cnt,name=('n/a','n/a',False,0,'<err>'); \
print('torch:', v); print('built_with_cuda:', cuda); print('cuda_available:', avail); print('device_count:', cnt); print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES')); print('device_name:', name)"

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

test:
	$(CONDA_RUN) env AIOBS_TEST_MODE=1 python -m pytest -q

test-v:
	$(CONDA_RUN) env AIOBS_TEST_MODE=1 python -m pytest -v

# Build in safe mode (does not touch real index_dir). Optionally set AIOBS_TEST_INDEX_DIR=/tmp/aiobs-test
build-test:
	$(CONDA_RUN) env AIOBS_TEST_MODE=1 python -m cli.aiobs build
