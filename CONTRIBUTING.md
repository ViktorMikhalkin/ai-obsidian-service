# Contributing

Thank you for contributing! This guide covers development workflow, testing, and code standards.

---

## Development Setup

### Prerequisites

- Python 3.11
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- For GPU: NVIDIA GPU with CUDA 12.1+

### Create Environment

**GPU (Recommended):**
```bash
make env-create-gpu
conda activate aiobs-gpu
make env-validate-gpu
```

**CPU:**
```bash
make env-create-cpu
conda activate aiobs-cpu
```

### Install Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files  # Test
```

---

## Development Workflow

1. **Create branch:**
   ```bash
   git checkout -b feat/your-feature
   ```

2. **Make changes** in `src/` or `tests/`

3. **Run checks:**
   ```bash
   make check  # lint + type + test
   ```

4. **Test manually:**
   ```bash
   make serve-gpu
   ```

5. **Commit:**
   ```bash
   git commit -m "feat(api): add feature"
   ```

6. **Create PR**

---

## Code Quality

### Linting

```bash
make lint   # Check
make fmt    # Auto-fix
```

### Type Checking

```bash
make type
```

All functions need type hints:
```python
from __future__ import annotations

def process(path: str) -> int:
    ...
```

### Style Guidelines

- 120 char line length
- Type hints required
- Docstrings for public APIs
- Follow PEP 8

---

## Testing

### Run Tests

```bash
make test              # Unit tests (fast)
make test-faiss-cpu    # FAISS CPU
make test-faiss-gpu    # FAISS GPU
make test-all          # Full suite
```

### Test Markers

```python
@pytest.mark.integration_cpu
def test_faiss():
    ...

@pytest.mark.integration_gpu
def test_gpu():
    ...
```

Run specific tests:
```bash
pytest -m "integration_cpu"
pytest -k "search"
pytest tests/test_file.py::test_func -v
```

### Writing Tests

```python
def test_embedder_batch():
    embedder = SentenceTransformersEmbedder()
    texts = ["hello", "world"]

    embeddings = embedder.embed_batch(texts)

    assert embeddings.shape == (2, embedder.dim)
```

---

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`

**Scopes:** `api`, `cli`, `index`, `parser`, `faiss`, `rag`, `deps`, `docs`, `tests`, `ci`, `repo`

**Examples:**
```bash
feat(api): add SSE streaming for index rebuild
fix(parser): handle empty PDF documents
perf(index): implement batch embedding
docs(readme): update API examples
```

**Breaking changes:**
```bash
feat(api)!: change response format

BREAKING CHANGE: /search now returns SearchResult object
```

---

## Pull Request Process

### Before Creating PR

```bash
make check  # Must pass
```

### PR Requirements

- Title follows Conventional Commits
- Description includes:
    - What changed and why
    - Link to related issues
    - Breaking changes (if any)
- Tests added for new features
- Documentation updated

### Review Process

- CI must pass
- At least one approval required
- Address review comments
- Squash commits (optional)

---

## API Testing

### Using curl

```bash
# Health
curl http://127.0.0.1:8000/health

# Info
curl http://127.0.0.1:8000/info | jq

# Rebuild (SSE)
curl -N -X POST http://127.0.0.1:8000/index/rebuild \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"root": "/path/to/vault"}'

# Search
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 5}' | jq
```

### Using HTTP Client

See `api-tests.http` for comprehensive examples.

Open in IntelliJ/PyCharm and click play button.

### Using Test Scripts

```bash
chmod +x sse_test_examples.sh
./sse_test_examples.sh
```

---

## Performance

### Profile Code

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Monitor GPU

```bash
watch -n 1 nvidia-smi
```

### Benchmark API

```bash
time curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 5}' -o /dev/null -s
```

---

## Common Tasks

### Add New Endpoint

1. Add to `src/ai_obsidian_service/api/app.py`
2. Add schemas to `schemas.py`
3. Add tests to `tests/api/`
4. Update `api-tests.http`
5. Run `make check`

### Add Document Parser

1. Create `src/ai_obsidian_service/parsers/parser_format.py`
2. Implement `DocumentParser` interface
3. Register in `di_selector.py`
4. Add tests

### Update Dependencies

```bash
# Edit environment file
vi environment.gpu.yml

# Recreate
make env-remove-gpu
make env-create-gpu
```

---

## Getting Help

- Check logs for error messages
- Review existing issues/PRs
- Ask in GitHub Discussions
- Check documentation: http://127.0.0.1:8000/docs

---

## Code of Conduct

Follow our [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful and professional.
