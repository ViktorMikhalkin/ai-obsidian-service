# syntax=docker/dockerfile:1.7

ARG FLAVOR=cpu

# CPU build: lightweight slim Python
FROM python:3.12-slim AS base-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget tini && \
    rm -rf /var/lib/apt/lists/*

# GPU build: proven PyTorch CUDA base (works reliably)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS base-gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini && \
    rm -rf /var/lib/apt/lists/*

# Select base based on FLAVOR
FROM base-${FLAVOR} AS final

WORKDIR /app

# ---- Install Python packages ----
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# For CPU: Set environment to force CPU-only packages
ENV CUDA_VISIBLE_DEVICES="" \
    FORCE_CUDA=0

# Install PyTorch (CPU-only, just torch for NLP)
# For GPU: PyTorch is already in base image, don't reinstall!
RUN if [ "${FLAVOR}" = "cpu" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu; \
    fi

# Install scientific stack (compatible with PyTorch)
# Force CPU-only numpy to avoid CUDA versions
RUN pip install --no-cache-dir \
      "numpy<2.0" \
      "scipy>=1.10" \
      "pandas>=2.0" \
      "scikit-learn>=1.3"

# Install FAISS CPU explicitly (no CUDA)
RUN if [ "${FLAVOR}" = "gpu" ]; then \
      pip install --no-cache-dir faiss-gpu; \
    else \
      pip install --no-cache-dir faiss-cpu==1.8.0; \
    fi

# Install transformers ecosystem WITHOUT torch dependencies
# (torch already installed above as CPU-only)
RUN if [ "${FLAVOR}" = "cpu" ]; then \
      pip install --no-cache-dir --no-deps \
        "transformers>=4.34,<5.0" \
        "huggingface-hub>=0.15.1" \
        "tokenizers>=0.13" \
        "safetensors>=0.3.1" \
        "sentence-transformers>=2.2,<3.0"; \
    else \
      pip install --no-cache-dir \
        "transformers>=4.34,<5.0" \
        "huggingface-hub>=0.15.1" \
        "sentence-transformers>=2.2,<3.0"; \
    fi

# Install missing dependencies that we skipped with --no-deps
RUN if [ "${FLAVOR}" = "cpu" ]; then \
      pip install --no-cache-dir \
        tqdm regex requests filelock packaging \
        "pyyaml>=5.1" "pillow>=8.0"; \
    fi

# Install API packages
RUN pip install --no-cache-dir \
      "fastapi>=0.104" \
      "uvicorn[standard]>=0.24" \
      "pydantic>=2"

# Install document processing
RUN pip install --no-cache-dir \
      "rank-bm25>=0.2.2" \
      "pypdf>=4.0" \
      "beautifulsoup4>=4.13" \
      "ebooklib>=0.18" \
      "python-frontmatter>=1.0.0" \
      "pymupdf>=1.24.0" \
      "tiktoken>=0.5.0" \
      "ocrmypdf>=16.11.0"

# Install dev/typing packages
RUN pip install --no-cache-dir \
      "respx>=0.20.0" \
      "types-PyYAML" \
      "types-requests" \
      "types-tqdm" \
      "types-aiofiles"

# Copy application
COPY . /app

# Verify PyTorch installation (especially for GPU)
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Environment
ENV PYTHONPATH=/app/src
ENV AI_OBS_CONFIG_PATH=/config/service_config.json

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "ai_obsidian_service.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "7200", "--timeout-graceful-shutdown", "30"]
