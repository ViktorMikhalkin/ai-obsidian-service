# AI ↔ Obsidian — Local Indexing & RAG Service

Local FastAPI service and CLI to index Obsidian Markdown notes and external PDF/EPUB libraries into a FAISS vector store, and to query them via REST or CLI. Optional GPU acceleration for embeddings and optional answer generation via Ollama.

---

## Contents
- What you get
- Prerequisites
- Setup (CPU/GPU)
- Configuration
- Usage (CLI and API)
- Development & Testing
- Troubleshooting
- Suggested healthchecks & tests

---

## What you get
- One searchable place for your electronic library (Markdown notes, PDFs, and EPUBs)
- Find the exact passage you need with citations (e.g., page numbers for PDFs, sections for EPUB/Markdown)
- See helpful bibliographic details with results (title, author when available, file path/location)
- Ask natural-language questions about your library and get concise answers with source quotes
- Private and fast: everything runs locally on your machine (no cloud required)
- Works with your Obsidian vault and additional folders you point to
- Rebuild the index anytime; check status at a glance
- Optional: accelerate indexing with your GPU
- Optional: richer answers using a local LLM via Ollama

---

## Prerequisites
- Conda or Mamba (Miniconda/Mambaforge recommended)
- Python 3.12 (will be created inside the conda env)
- Poppler (for PDF parsing, only if you index PDFs)
    - Linux: `sudo apt install -y poppler-utils`
    - macOS: `brew install poppler`
- NVIDIA GPU (optional, only if you want GPU acceleration for embeddings)

---

## Setup

### CPU setup

```shell
# 1) Create env and install dependencies
make setup-cpu      # = env-cpu + install + install-test-deps (pytest)

# 2) Create and configure config
cp config.yaml.example config.yaml
# edit vault_path and library_paths

# 3) Quick smoke test (safe mode; won't touch your real index/)
AIOBS_TEST_MODE=1 make build-test
AIOBS_TEST_MODE=1 make status

# 4) Full index build
make build
make status

# 5) Run API
make serve
# Open http://127.0.0.1:8000/docs
```

### GPU setup

```shell
# 1) Create GPU env with CUDA 12.1 PyTorch
make ENV=aiobs-gpu setup-gpu

# 2) Verify CUDA
make ENV=aiobs-gpu check-cuda   # expect cuda_available: True

# 3) Set config.yaml for GPU (embeddings.device: "cuda"; reduce batch_size if OOM)
# 4) Build index on GPU
make ENV=aiobs-gpu build
make ENV=aiobs-gpu status

# 5) Run API
make ENV=aiobs-gpu serve
```

### Core Makefile commands

- `make setup-cpu` — CPU env + deps + pytest
- `make setup-gpu` — GPU env + deps + pytest
- `make build` — index data (reads config.yaml)
- `make build-test` — index in safe test mode (AIOBS_TEST_MODE=1), won't touch real index/
- `make status` — show number of chunks in index
- `make serve` — run API with autoreload
- `make check-cuda` — CUDA diagnostics (for GPU env)
- `make clean` — remove index/

### Choosing env name and Python version
- Defaults: ENV=aiobs-cpu and PY=3.12.
- Override on the fly:
    - `make ENV=aiobs-gpu build`
    - `make PY=3.12 ENV=aiobs-gpu env-gpu`

### Safe mode (for tests/demos)

- Use `AIOBS_TEST_MODE=1` to write the index into a temporary, safe directory:
    - `AIOBS_TEST_MODE=1 make build-test`
    - You can force a specific location:
        - `AIOBS_TEST_INDEX_DIR=/tmp/aiobs-test-index AIOBS_TEST_MODE=1 make build-test`

## Development

### Available Make Commands

| Command | Description |
|---------|-------------|
| `make setup-cpu` | Create CPU environment with dependencies |
| `make setup-gpu` | Create GPU environment with CUDA support |
| `make install` | Install Python dependencies |
| `make build` | Build/rebuild the document index |
| `make build-test` | Build in safe test mode (temporary index) |
| `make status` | Show index statistics |
| `make serve` | Start API server with auto-reload |
| `make check-cuda` | Verify CUDA availability |
| `make clean` | Remove existing index |

### Environment Variables

- `ENV`: Choose conda environment (default: `aiobs-cpu`)
- `PY`: Python version (default: `3.12`)
- `AIOBS_TEST_MODE`: Use temporary index location (set to `1`)
- `AIOBS_TEST_INDEX_DIR`: Custom test index directory

### Examples

```bash
# Use GPU environment
make ENV=aiobs-gpu build

# Custom Python version and environment
make PY=3.12 ENV=my-custom-env setup-cpu

# Test mode with custom location
AIOBS_TEST_INDEX_DIR=/tmp/my-test AIOBS_TEST_MODE=1 make build-test
```

### Testing

Run the test suite:
```bash
# Install test dependencies (included in setup commands)
make install-test-deps

# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=src tests/
```

## Troubleshooting

### Common Issues

#### `pytest: not found`
**Cause**: Test dependencies not installed  
**Solution**: Run `make install-test-deps` or `make setup-cpu/setup-gpu`

#### `CUDA unavailable`
**Cause**: GPU drivers or CUDA toolkit issues  
**Solution**:
1. Check GPU driver: `nvidia-smi`
2. Verify CUDA: `make ENV=aiobs-gpu check-cuda`
3. If OOM errors: reduce `embeddings.batch_size` in config.yaml

#### `PDFs not parsed`
**Cause**: Poppler not installed  
**Solution**: Install Poppler (see [Prerequisites](#prerequisites))

#### `Indexing is slow`
**Causes & Solutions**:
- **Too many files**: Narrow `include_globs` patterns in config.yaml
- **Large chunks**: Reduce `chunk.target_tokens` (try 200)
- **CPU bottleneck**: Use GPU setup or increase `embeddings.batch_size` (64-128 for GPU)

#### `Import errors or module not found`
**Cause**: Wrong conda environment activated  
**Solution**: Activate the correct environment:
```bash
conda activate aiobs-cpu
# or
conda activate aiobs-gpu
```

### Performance Tuning

#### For Large Libraries (10k+ documents)
```yaml
# config.yaml optimizations
chunk:
  target_tokens: 200      # Smaller chunks = faster indexing
  
embeddings:
  batch_size: 128         # Higher batch size for GPU
  device: "cuda"          # Use GPU if available

include_globs:
  - "**/*.pdf"            # Be selective with file types
  # - "**/*.md"           # Comment out unused patterns
```

#### Memory Usage
- **High RAM usage**: Reduce `embeddings.batch_size`
- **GPU OOM**: Lower `embeddings.batch_size` to 16-32
- **Disk space**: Run `make clean` periodically to remove old indexes

### Getting Help

1. **Check logs**: API server logs show detailed error information
2. **Test mode**: Use `AIOBS_TEST_MODE=1` for safe experimentation
3. **Minimal test**: Start with a small folder (1-3 PDFs) to verify setup
4. **Issue reporting**: Include your config.yaml (without sensitive paths) and error logs

### 🔧 Example Debugging Session

```bash
# 1. Start with minimal test
echo "include_globs: ['**/*.pdf']" > test-config.yaml
echo "library_paths: ['/path/to/small/pdf/folder']" >> test-config.yaml

# 2. Test in safe mode
AIOBS_TEST_MODE=1 AIOBS_CONFIG=test-config.yaml make build-test

# 3. Check what was indexed
AIOBS_TEST_MODE=1 make status

# 4. If successful, try full build
make build && make serve
```

See CONTRIBUTING.md for pre-commit hooks and development guidelines.