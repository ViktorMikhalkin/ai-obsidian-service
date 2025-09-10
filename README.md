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
- Optional reranking / answer generation via **Ollama**.
- REST API for plugin integration + CLI for standalone use.

---

## Installation

### Requirements
- Python 3.10+
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended)
- Poppler (`pdftotext`) for PDF parsing
  ```bash
  # Linux (Debian/Ubuntu)
  sudo apt install poppler-utils
  # macOS
  brew install poppler
  # conda
  conda install -c conda-forge poppler
  ```

### Setup with conda
```bash
cd ai-obsidian-service-0.1.0

# create env from environment.yml
conda env create -n ai-obsidian -f environment.yml
conda activate ai-obsidian

# OR manual setup:
# conda create -n ai-obsidian python=3.11 -y
# conda activate ai-obsidian
# conda install -c conda-forge faiss-cpu poppler
# pip install -r requirements.txt
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

### 1. Build index
```bash
python -m cli.aiobs build
```

### 2. Run API
```bash
uvicorn indexer.app:app --host 127.0.0.1 --port 8000 --reload
```

Check health:
```bash
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

### 3. Search
```bash
curl -s http://127.0.0.1:8000/search \
  -H "content-type: application/json" \
  -d '{"query":"transactional outbox pattern","top_k":5}'
```

### 4. Answer with citations (requires Ollama)
```bash
curl -s http://127.0.0.1:8000/answer \
  -H "content-type: application/json" \
  -d '{"query":"how to implement transactional outbox in Spring?","top_k":8}'
```

### 5. CLI queries
```bash
python -m cli.aiobs search "event sourcing"
```

---

## Development

Run tests:
```bash
pytest tests/
```


### Index management

- Stats
```bash
curl -s http://127.0.0.1:8000/index/stats | jq .
```

- Rebuild (fire-and-return)
```bash
curl -s -X POST http://127.0.0.1:8000/index/rebuild -H "content-type: application/json" -d '{"timeout_sec":0}' | jq .
```


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

## Contributing (Commit Hooks)

This repo uses **husky + commitlint** to validate commit messages (Conventional Commits).
To enable locally:

```bash
# one-time
npm install

# test the hook manually
echo "feat: test message" | npx commitlint
```


---

## 🤝 Contributing

We welcome community contributions!  
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on development workflow, commit conventions, and pull requests.  

- Use **Conventional Commits** for commit messages (`type(scope): message`).  
- Run tests locally with `pytest -v`.  
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

## Conda / Mamba setup (Python 3.12/3.13)

**Full environment (FAISS 1.12.0 + CPU PyTorch + embeddings):**
```bash
mamba env create -f environment.yml    # or: conda env create -f environment.yml
conda activate aiobs

cp config.yaml.example config.yaml
# edit vault_path and library_paths

python -m cli.aiobs build
uvicorn indexer.app:app --reload --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000/docs
```

**Core environment (BM25-only, quick smoke test):**
```bash
mamba env create -f environment-core.yml
conda activate aiobs-core

cp config.yaml.example config.yaml
python -m cli.aiobs build
uvicorn indexer.app:app --reload --host 127.0.0.1 --port 8000
```

> Notes:
> - On Python 3.12/3.13, FAISS via pip may be unavailable; conda is reliable.
> - BM25 fallback works without embeddings as long as `index/index.jsonl` exists.
> - Dockerfile can be used later for releases.

---

## Quick Start (Conda) — Updated

```bash
# Full environment (FAISS + embeddings)
mamba env create -f environment.yml
conda activate aiobs

# (Alternative) Minimal BM25-only API:
# mamba env create -f environment-core.yml
# conda activate aiobs-core

# Configure
cp config.yaml.example config.yaml
# edit: vault_path, library_paths, index_dir

# Build index (creates index/faiss.index, index/index.jsonl, index/dim.txt)
python -m cli.aiobs build

# Run API
uvicorn indexer.app:app --reload --host 127.0.0.1 --port 8000

# Smoke tests
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/search -H 'Content-Type: application/json' -d '{"query":"java performance","top_k":5}'
```

## Troubleshooting

- **Empty hits**: ensure `index/faiss.index`, `index/index.jsonl`, `index/dim.txt` exist. Re-run `python -m cli.aiobs build` if needed.
- **Path mismatch**: API reads `index_dir` from `config.yaml`. Keep paths consistent.
- **FAISS via pip on 3.13**: use Conda (`environment.yml`) for reliable FAISS installation.
- **Minimal mode**: `environment-core.yml` works without embeddings; still needs `index/index.jsonl`.


## GPU Usage

The indexing pipeline has two main compute-heavy parts:

1. **Embeddings (GPU-accelerated)**
    - Controlled via `config.yaml`:
      ```yaml
      embeddings:
        model: "intfloat/multilingual-e5-small"
        device: "cuda"     # use "cpu" if no GPU
        batch_size: 128    # reduce to 96 or 64 if you hit CUDA OOM
      ```
    - When `device: cuda` is set, embeddings are computed directly on your NVIDIA GPU.
    - This stage is the bottleneck of indexing, and GPU acceleration typically yields a **10×–30× speedup** compared to CPU-only runs.
    - You can monitor usage with:
      ```bash
      nvidia-smi -l 2
      ```
      Expect GPU utilization >50% and memory usage in the 1–2 GB range during embedding.

2. **FAISS Index (CPU by default)**
    - Controlled via `config.yaml`:
      ```yaml
      faiss:
        search_on: "cpu"
      ```
    - By default, the FAISS index is built and stored on CPU. This keeps the index portable and avoids tying it to one specific GPU.
    - At query time, you can set `search_on: gpu` to run nearest-neighbor search on GPU for faster queries. The index is still stored on disk in CPU format.

### Performance Expectations

- On an RTX 1000 Ada (6 GB VRAM):
    - **GPU (batch_size=128):** ~5–10k embeddings/sec → full library (50–60k chunks) in a few minutes.
    - **CPU:** ~500–800 embeddings/sec → same workload in 1–2 hours.
- Adjust `batch_size` if you see CUDA out-of-memory (OOM). Lower values (96 or 64) trade some speed for lower memory use.

### Quick Test

```bash
make check-cuda   # verify CUDA is available in your environment
make build        # run indexing with GPU acceleration
