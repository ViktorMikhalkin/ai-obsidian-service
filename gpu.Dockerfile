# syntax=docker/dockerfile:1.7
# ---------- Stage 1: build venv with pip only ----------
FROM ubuntu:24.04 AS builder

ARG PY_VER=3.11
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu129
ENV VENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-c"]

# Minimal system deps for Python + strip
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl python3 python3-venv python3-pip binutils; \
    rm -rf /var/lib/apt/lists/*

# Create venv and upgrade tooling
RUN python3 -m venv "$VENV"
RUN python -m pip install -U pip setuptools wheel

# --- GPU stack (pip-only) ---
RUN python -m pip install --no-cache-dir \
      --index-url $TORCH_INDEX \
      torch==2.8.0

RUN python -m pip install --no-cache-dir \
      faiss-gpu-cu12

# --- App & deps ---
WORKDIR /opt/app
COPY . /opt/app
RUN python -m pip install --no-cache-dir /opt/app && \
    python -m pip install --no-cache-dir \
      fastapi>=0.104 uvicorn>=0.24 pydantic>=2 typer>=0.9 aiofiles>=23 python-multipart>=0.0.6 \
      jinja2>=3.1 python-dotenv>=1.0 pyyaml>=6.0 httpx>=0.25 requests>=2.31 lxml>=4.9.0 \
      pymupdf>=1.24.0 ocrmypdf>=16.11.0 beautifulsoup4>=4.13 ebooklib>=0.18 python-frontmatter>=1.0.0 \
      "sentence-transformers>=2.2,<3.0" rank-bm25>=0.2.2 pypdf>=4.0 tiktoken>=0.5.0

# Quick check
RUN python - <<'PY'
import torch, faiss
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
print("faiss:", getattr(faiss, "__version__", "n/a"))
PY

# Strip .so symbols + clean pip cache
RUN find "$VENV/lib" -type f -name "*.so*" -exec strip --strip-unneeded {} \; || true && \
    rm -rf /root/.cache/pip

# ---------- Stage 2: runtime with multilingual OCR ----------
FROM ubuntu:24.04

ENV VENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/app/src \
    AI_OBS_CONFIG_PATH=/config/service_config.json \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/app
SHELL ["/bin/bash", "-c"]

# Multilingual OCR toolchain (ENG+RUS+UKR) + tini
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends software-properties-common; \
    add-apt-repository -y universe; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      tini \
      tesseract-ocr \
      tesseract-ocr-eng tesseract-ocr-rus tesseract-ocr-ukr \
      qpdf ghostscript poppler-utils \
      pngquant unpaper \
      ca-certificates \
      wget; \
    rm -rf /var/lib/apt/lists/*

# Bring venv + app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/app /opt/app

# Final trim
RUN find "$VENV/lib" -type f -name "*.a" -delete || true && rm -rf /root/.cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
    CMD wget -qO- http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "ai_obsidian_service.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "7200", "--timeout-graceful-shutdown", "30"]
