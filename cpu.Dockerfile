# syntax=docker/dockerfile:1.7
# AI Obsidian Service — CPU (robust: conda-only solve, pip second, conda-pack)

############################
# 1) Builder: create & pack
############################
FROM mambaorg/micromamba:2.3.2-debian13-slim AS builder

ENV MAMBA_NO_BANNER=true \
    MAMBA_ALWAYS_YES=true \
    MICROMAMBA_LOG_LEVEL=info

WORKDIR /build
USER root

# 1) Bring environment spec
COPY environment.cpu.prod.yml /tmp/environment.yml

# 2) Normalize BOM/CRLF
RUN awk 'NR==1{sub(/^\xef\xbb\xbf/,"")} {sub(/\r$/,"")}1' /tmp/environment.yml > /tmp/env.yml && mv /tmp/env.yml /tmp/environment.yml

# 3) Write awk to strip the `- pip:` block (indentation-aware)
RUN cat >/tmp/strip_pip.awk <<'AWK'
function indent(s){ m=match(s,/[^ ]/); return m?m-1:length(s) }
BEGIN{ inpip=0; pipindent=0 }
{
  if ($0 ~ /^[[:space:]]*-[[:space:]]+pip[[:space:]]*:[[:space:]]*$/) {
    inpip=1; pipindent=indent($0); next
  }
  if (inpip) {
    if ($0 ~ /^[[:space:]]*-[[:space:]]/ && indent($0) <= pipindent) {
      inpip=0; print; next
    } else next
  }
  print
}
AWK

# 4) Produce conda-only YAML and show its head
RUN awk -f /tmp/strip_pip.awk /tmp/environment.yml > /tmp/environment.conda.yml && \
    echo "----- conda YAML (head) -----" && nl -ba /tmp/environment.conda.yml | head -n 120 || true && echo "----------------"

# 5) Create conda env by explicit prefix (conda-only)
RUN micromamba config set channel_priority strict && \
    micromamba create -p /opt/conda/envs/aiobs-cpu -f /tmp/environment.conda.yml && \
    test -x /opt/conda/envs/aiobs-cpu/bin/python

# 6) Install pip dependencies
#    6.1) Official CPU install for PyTorch (exactly as in the docs)
ENV PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore PYTHONDONTWRITEBYTECODE=1
RUN /opt/conda/envs/aiobs-cpu/bin/python -m pip install --upgrade pip && \
    /opt/conda/envs/aiobs-cpu/bin/pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      torch

#    6.2) Install the rest of your pip deps (without torch)
RUN cat >/tmp/requirements-pip.txt <<'REQS'
--prefer-binary
sentence-transformers>=2.2,<3.0
faiss-cpu==1.12.0
rank-bm25>=0.2.2
pypdf>=4.0
beautifulsoup4>=4.13
ebooklib>=0.18
python-frontmatter>=1.0.0
pymupdf>=1.24.0
tiktoken>=0.5.0
ocrmypdf>=16.11.0
REQS
RUN /opt/conda/envs/aiobs-cpu/bin/pip install -r /tmp/requirements-pip.txt

# 7) Slim & pack by prefix
RUN micromamba install -p /opt/conda/envs/aiobs-cpu conda-pack -c conda-forge && \
    find /opt/conda/envs/aiobs-cpu -type f \( -name '*.a' -o -name '*.pyc' -o -name '*.js.map' \) -delete && \
    micromamba clean --all --yes && rm -rf /opt/conda/pkgs && \
    /opt/conda/envs/aiobs-cpu/bin/conda-pack \
      -p /opt/conda/envs/aiobs-cpu \
      -o /tmp/aiobs-cpu.tar.gz \
      --ignore-missing-files

############################
# 8) Runtime: tiny & fast
############################
FROM debian:bookworm-slim

# Minimal runtime deps + tools for ocrmypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini ca-certificates bzip2 xz-utils \
    ghostscript qpdf tesseract-ocr pngquant \
    && rm -rf /var/lib/apt/lists/*

# copy the packed conda env from builder
COPY --from=builder /tmp/aiobs-cpu.tar.gz /opt/aiobs-cpu.tar.gz

# Unpack the prebuilt environment once at build time
RUN mkdir -p /opt/env \
 && tar -xzf /opt/aiobs-cpu.tar.gz -C /opt/env \
 && rm /opt/aiobs-cpu.tar.gz \
 && /opt/env/bin/python /opt/env/bin/conda-unpack \
 && /opt/env/bin/python -c "import sys; print('Python OK:', sys.version)"


# Make the env active by default
ENV PATH="/opt/env/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/env/lib:${LD_LIBRARY_PATH}" \
    CONDA_DEFAULT_ENV="aiobs-cpu" \
    PYTHONPATH=/app/src \
    AI_OBS_CONFIG_PATH=/config/service_config.json

WORKDIR /app
COPY . /app

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "ai_obsidian_service.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "7200", "--timeout-graceful-shutdown", "30"]
