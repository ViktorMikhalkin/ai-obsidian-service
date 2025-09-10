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

ENV ?= aiobs-gpu
PY  ?= 3.12

# Channels are set per-env to avoid global config drift
CHANNELS = -c pytorch -c nvidia

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

# ---- Project deps (pip) ----
install:
	conda run -n $(ENV) pip install -U -r requirements.txt

# ---- Diagnostics ----
check-cuda:
	@echo ">> Checking CUDA in env $(ENV)"
	conda run -n $(ENV) python - <<'PY'\
import torch, os; \
print("torch:", torch.__version__); \
print("built_with_cuda:", torch.version.cuda); \
print("cuda_available:", torch.cuda.is_available()); \
print("device_count:", torch.cuda.device_count()); \
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")); \
print("device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "<none>")\
PY

# ---- App tasks ----
build:
	conda run -n $(ENV) python -m cli.aiobs build

status:
	conda run -n $(ENV) python -m cli.aiobs status

serve:
	conda run -n $(ENV) python -m cli.aiobs serve --host 127.0.0.1 --port 8000

serve-no-reload:
	conda run -n $(ENV) python - <<'PY'\
import uvicorn; uvicorn.run("indexer.app:app", host="127.0.0.1", port=8000, reload=False)\
PY

clean:
	rm -rf index
