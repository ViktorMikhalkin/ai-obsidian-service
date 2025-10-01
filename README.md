# AI ↔ Obsidian — Python Service

FastAPI-based local indexing and RAG service for Obsidian integration with real-time SSE progress streaming.

Parses Markdown, PDF, and EPUB documents, builds a FAISS vector index with GPU acceleration, and exposes REST APIs for semantic search and retrieval-augmented generation.

---

## Features

- Document parsing: Markdown, PDF (Poppler), EPUB
- Vector search with FAISS GPU acceleration
- Real-time indexing with SSE streaming progress
- RAG with Ollama integration
- REST API for Obsidian plugin

---

## Installation

### Prerequisites

- Python 3.11 (3.13 not yet supported)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- For GPU: NVIDIA GPU with CUDA 12.1+
- For PDF: Poppler (`sudo apt install poppler-utils`)

### Setup

**GPU (Recommended):**
```bash
git clone https://github.com/yourusername/ai-obsidian-service.git
cd ai-obsidian-service
make env-create-gpu
conda activate aiobs-gpu
```

**CPU:**
```bash
make env-create-cpu
conda activate aiobs-cpu
```

### Configuration

Create `.env`:
```bash
VECTOR_STORE_BACKEND=faiss
INDEX_DIR=./.index
EMBEDDER_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: Ollama for RAG
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

Load: `set -a && source .env && set +a`

---

## Usage

### Start Server

```bash
make serve-gpu  # or serve-cpu
# http://127.0.0.1:8000
```

### API Examples

**Index information:**
```bash
curl http://127.0.0.1:8000/info | jq
```

**Rebuild index (SSE streaming):**
```bash
curl -N -X POST http://127.0.0.1:8000/index/rebuild \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"root": "/path/to/vault"}'
```

**Search:**
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 5}' | jq
```

**Generate answer:**
```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 5}' | jq
```

**Documentation:** http://127.0.0.1:8000/docs

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code standards, and testing.

Quick start:
```bash
make env-create-gpu
conda activate aiobs-gpu
make check  # lint + type + test
```

**Commit convention:** [Conventional Commits](https://www.conventionalcommits.org/)
```bash
feat(api): add SSE streaming
fix(parser): handle empty PDFs
```

---

## License

Apache License 2.0 - See [LICENSE](LICENSE)

---

## Links

- Documentation: http://127.0.0.1:8000/docs
- Issues: [GitHub Issues](https://github.com/yourusername/ai-obsidian-service/issues)
- Security: [SECURITY.md](SECURITY.md)
