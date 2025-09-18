# AI ↔ Obsidian — Python Service

FastAPI-based local indexing and RAG service for Obsidian integration.
Parses Markdown notes and external PDF/EPUB libraries, builds a FAISS vector index,
and exposes `/search` and `/answer` APIs for the Obsidian plugin.
Includes a CLI for manual indexing and queries.

---

## Features
- Parse & index **Markdown** notes (frontmatter aware).
- Parse & index **PDF** via `pdftotext` (Poppler).
- Optional **EPUB** parsing.
- Vector store with **FAISS** (cosine similarity).
- **GPU acceleration** support with CUDA/PyTorch and faiss-gpu.
- Optional reranking / answer generation via **Ollama**.
- REST API for plugin integration + CLI for standalone use.

---

## Installation

### Requirements
- Python 3.12+ (recommended for GPU support)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/) (recommended)
- **For GPU support**: NVIDIA GPU with CUDA 12.1+ drivers
- Poppler (`pdftotext`) for PDF parsing
  ```bash
  # Linux (Debian/Ubuntu)
  sudo apt install poppler-utils
  # macOS
  brew install poppler
  # conda (included in environment files)
  conda install -c conda-forge poppler
  ```

### Quick Setup with Makefile (Recommended)

**GPU Environment (Recommended for performance):**
```bash
cd ai-obsidian-service
make quickstart-gpu
```

**CPU-only Environment (fallback option):**
```bash
cd ai-obsidian-service
make quickstart-cpu
```

### Manual Setup with Conda

**GPU-accelerated environment (Recommended for best performance):**
```bash
cd ai-obsidian-service

# Create and verify GPU environment
make setup-gpu
make check-gpu  # Verify CUDA and faiss-gpu work

# Expected output:
# ✅ CUDA setup OK!
# ✅ faiss-gpu setup OK!
```

**CPU-only environment (Fallback option):**
```bash
cd ai-obsidian-service

# Create CPU environment
make setup-cpu
```

**Core environment (BM25-only, minimal dependencies):**
```bash
conda env create -f environment-core.yml -n aiobs-core
conda activate aiobs-core
```

### GPU Setup Verification

After setting up the GPU environment, verify everything works:

```bash
# Check CUDA and PyTorch
make check-cuda

# Check faiss-gpu
make check-faiss

# Check both
make check-gpu
```

Expected output:
```
✅ CUDA setup OK!
torch: 2.5.1
cuda_available: True
device_count: 1
device_name[0]: NVIDIA RTX 1000 Ada Generation Laptop GPU

✅ faiss-gpu setup OK!
faiss-gpu available: True
GPU resources initialized successfully
```

---

## Configuration

Edit `config.yaml`:

```yaml
vault_path: "/path/to/your/obsidian-vault"
library_paths:
  - "/path/to/pdf/library"
  - "/path/to/epub/library"
include_globs:
  - "**/*.md"
  - "**/*.pdf"
  - "**/*.epub"

embeddings:
  provider: "local"
  model: "intfloat/multilingual-e5-small"
  # GPU acceleration will be used automatically if available

llm:
  provider: "ollama"
  model: "qwen2.5:7b-instruct"
  base_url: "http://127.0.0.1:11434"

rag:
  mode: "generative"
  top_k: 8
  rerank: false
```

---

## Usage

### Using Makefile (Recommended)

```bash
# Build index (GPU-accelerated by default)
make build

# Check index status
make status

# Run API server
make serve
```

### Manual Commands

```bash
# 1. Build index
python -m ai_obsidian_service.cli.aiobs build

# 2. Run API
uvicorn ai_obsidian_service.indexer.app:app --host 127.0.0.1 --port 8000 --reload
```

### API Endpoints

Check health:
```bash
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

Search documents:
```bash
curl -s http://127.0.0.1:8000/search \
  -H "content-type: application/json" \
  -d '{"query":"transactional outbox pattern","top_k":5}'
```

Generate answers with citations (requires Ollama):
```bash
curl -s http://127.0.0.1:8000/answer \
  -H "content-type: application/json" \
  -d '{"query":"how to implement transactional outbox in Spring?","top_k":8}'
```

### CLI Usage

```bash
# Search via CLI
python -m ai_obsidian_service.cli.aiobs search "event sourcing"

# Get index statistics
make status
```

---

## Development

### Running Tests
```bash
# Install test dependencies (if not already done)
make install-test-deps

# Run tests
make test

# Run tests with coverage
make test-coverage
```

### Index Management

Get index statistics:
```bash
curl -s http://127.0.0.1:8000/index/stats | jq .
```

Rebuild index (async):
```bash
curl -s -X POST http://127.0.0.1:8000/index/rebuild \
  -H "content-type: application/json" \
  -d '{"timeout_sec":0}' | jq .
```

### Environment Management

```bash
# View environment info
make info

# Clean index files
make clean

# Remove conda environment completely
make clean-env

# Reinstall package after code changes
make reinstall
```

---

## Performance Notes

### GPU Acceleration

With GPU support enabled, you'll see significant performance improvements:

- **Embedding generation**: 5-10x faster with CUDA
- **Vector similarity search**: 3-5x faster with faiss-gpu
- **Batch processing**: Scales better with larger document sets

### Hardware Recommendations

- **CPU-only**: Any modern CPU, 8GB+ RAM
- **GPU-accelerated**:
    - NVIDIA GPU with 4GB+ VRAM (RTX 3060+, RTX 4060+, or better)
    - 16GB+ system RAM for large document collections
    - CUDA 12.1+ drivers

---

## API Documentation

- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- Static OpenAPI spec: [api/openapi.json](api/openapi.json)

To regenerate `api/openapi.json` from code:
```bash
python scripts/export_openapi.py
```

---

## Troubleshooting

### Common Issues

**Empty search results:**
- Ensure index files exist: `make status`
- Re-run `make build` if index files are missing

**GPU not detected:**
```bash
# Verify GPU setup
make check-gpu

# If CUDA not available, check drivers:
nvidia-smi

# Recreate GPU environment if needed:
make clean-env
make setup-gpu
```

**Environment conflicts:**
```bash
# Remove problematic environment and recreate
make clean-env ENV=problematic-env-name
make setup-gpu  # or setup-cpu
```

**Performance issues:**
- Use GPU environment for large document collections (>1k docs)
- Monitor memory usage with large PDF collections
- For CPU-only systems, consider smaller batch sizes in config

### Getting Help

- Check the [troubleshooting section](#troubleshooting) above
- Review logs when running `make serve` for detailed error messages
- Use `make info` to verify environment setup
- Test with minimal config first, then add complexity

---

## Contributing (Commit Hooks)

This repo uses **husky + commitlint** to validate commit messages (Conventional Commits).
To enable locally:

```bash
# one-time setup
npm install

# test the hook manually
echo "feat: test message" | npx commitlint
```

---

## 🤝 Contributing

We welcome community contributions!
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on development workflow, commit conventions, and pull requests.

- Use **Conventional Commits** for commit messages (`type(scope): message`).
- Run tests locally with `make test`.
- Regenerate the OpenAPI schema with `python scripts/export_openapi.py`.

All contributions are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 🐛 Issues & Feature Requests

If you find a bug or have an idea for improvement, please open an issue:

- 🐛 [Bug report](.github/ISSUE_TEMPLATE/bug_report.yml)
- ✨ [Feature request](.github/ISSUE_TEMPLATE/feature_request.yml)
- 🧩 [Task](.github/ISSUE_TEMPLATE/task.yml)

---

## 📦 Releases

This project uses [release-please](https://github.com/googleapis/release-please) for automated versioning and changelog generation.
All versions are tagged with annotated git tags and published as GitHub Releases.

---

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).
You are free to use, modify, and distribute the code, provided that you include the license text in any copies.

---

## 🔒 Security

If you discover a vulnerability, please **do not open a public issue**.
Instead, review our [Security Policy](SECURITY.md) for instructions on responsible disclosure.
