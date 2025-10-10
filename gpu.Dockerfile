# syntax=docker/dockerfile:1.7
# Variant B: Multi-stage для реального уменьшения размера

############################
# Stage 1: Builder - установка и очистка
############################
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS builder

# System tools
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget \
    && rm -rf /var/lib/apt/lists/*

# Ускорение conda
RUN conda install -y conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda config --set channel_priority strict

# Установка ТОЛЬКО faiss-gpu (PyTorch УЖЕ есть!)
RUN conda install -y -c pytorch -c nvidia -c conda-forge \
      faiss-gpu=1.12.0 && \
    conda clean -y --all

# Остальные зависимости через pip
RUN pip install --no-cache-dir --prefer-binary \
      "fastapi>=0.104" \
      "uvicorn>=0.24" \
      "pydantic>=2.0" \
      "httpx>=0.26" \
      "sentence-transformers>=2.2,<3.0" \
      "rank-bm25>=0.2.2" \
      "pypdf>=4.0" \
      "beautifulsoup4>=4.13" \
      "ebooklib>=0.18" \
      "python-frontmatter>=1.0.0" \
      "pymupdf>=1.24.0" \
      "tiktoken>=0.5.0" \
      "ocrmypdf>=16.11.0"

# Удаление ненужных модулей PyTorch
RUN pip uninstall -y torchaudio torchvision triton 2>/dev/null || true && \
    rm -rf /opt/conda/lib/python3.11/site-packages/torchaudio* \
           /opt/conda/lib/python3.11/site-packages/torchvision* \
           /opt/conda/lib/python3.11/site-packages/triton* \
           /opt/conda/lib/python3.11/site-packages/torch/test \
           /opt/conda/lib/python3.11/site-packages/torch/include

# Агрессивная чистка
RUN conda clean -y --all && \
    rm -rf /opt/conda/pkgs/* \
           /opt/conda/lib/tcl* \
           /opt/conda/lib/tk* \
           /opt/conda/lib/sqlite* \
           /opt/conda/share/terminfo \
           /opt/conda/share/doc \
           /opt/conda/share/man \
           /root/.conda \
           /root/.cache

RUN find /opt/conda -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/conda -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/conda -type f -name "*.pyc" -delete && \
    find /opt/conda -type f -name "*.pyo" -delete && \
    find /opt/conda -type f -name "*.a" -delete


############################
# Stage 2: Runtime - только финальное окружение
############################
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# System tools + OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
      tini \
      ca-certificates \
      wget \
      # OCR tools \
      ghostscript \
      qpdf \
      tesseract-ocr \
      pngquant \
    && rm -rf /var/lib/apt/lists/*

# Копируем ТОЛЬКО очищенное окружение conda
COPY --from=builder /opt/conda /opt/conda

# Environment
ENV PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/app/src \
    AI_OBS_CONFIG_PATH=/config/service_config.json \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/conda/lib:${LD_LIBRARY_PATH}

WORKDIR /app
COPY . /app

# Проверка
RUN python -c "import torch; print('✓ PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())" && \
    python -c "import faiss; print('✓ FAISS:', faiss.__version__)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
    CMD wget -qO- http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "ai_obsidian_service.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "7200", "--timeout-graceful-shutdown", "30"]
