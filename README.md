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
