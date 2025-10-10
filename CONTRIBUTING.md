# Contributing

Thank you for contributing! This guide covers development workflow, testing, code standards, and CI/CD processes.

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

## CI/CD Pipeline

### Workflows

| Workflow | Trigger | Duration | Purpose |
|----------|---------|----------|---------|
| **PR Title Check** | PR events | 10s | Enforce `type(scope): subject` |
| **Quality & Unit Tests** | PR + main push | ~7 min | Lint, type, unit tests |
| **Integration Tests** | main push | ~10 min | Real FAISS + embeddings |
| **E2E Tests** | Manual only | ~15 min | Full system |
| **Release** | main push | 30s | Creates Release PR |
| **Docker Publish** | Release published | ~15 min | CPU + GPU images |

### Release Flow

```
Feature PR → tests pass → merge main
    ↓
Quality & Integration both pass (~10 min)
    ↓
Release PR created (if feat/fix commits)
    ↓
Merge Release PR → Tag + Release + Docker images
```

### Local CI Simulation

```bash
make check          # What PR checks run
make test-faiss-cpu # What integration runs
make test-e2e       # What e2e runs
```

### Makefile vs CI Decision Matrix

| Task | Use Makefile? | In CI? | Reason |
|------|--------------|--------|---------|
| Running tests | ✅ Yes | `make test` | Keep local/CI identical |
| Linting | ✅ Yes | `make lint` | Same rules everywhere |
| Type checking | ✅ Yes | `make type` | Consistent checks |
| Format checking | ✅ Yes | `make lint` | Ruff handles both |
| Environment validation | ✅ Yes | `make env-validate-cpu` | Useful locally too |
| Installing Conda env | ❌ No | GitHub Actions | Better caching in CI |
| Checking out code | ❌ No | GitHub Actions | Native functionality |
| Uploading artifacts | ❌ No | GitHub Actions | CI-specific |

**Philosophy:** Makefile contains **logic** (what to test/check), workflows contain **infrastructure** (how to set up environment).

### Triggering Manual Workflows

**Integration tests:**
```bash
# Via GitHub UI
# Actions tab → "Integration Tests (FAISS CPU)" → Run workflow

# Or via gh CLI
gh workflow run integration-tests-cpu.yml
```

**E2E tests:**
```bash
gh workflow run e2e-tests.yml
```

---

## Code Quality

### Linting

```bash
make lint   # Check (format + linting)
make fmt    # Auto-fix
```

**What CI runs:**
```bash
python -m ruff format --check .  # No auto-fix in CI
python -m ruff check .
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

- 88 char line length (Black/Ruff default)
- Type hints required
- Docstrings for public APIs
- Follow PEP 8

---

## Testing

### Test Categories

We use **pytest markers** to categorize tests:

| Marker | What it tests | Run locally | Run in CI |
|--------|---------------|-------------|-----------|
| `unit` (default) | Fast tests, mocks/fakes | `make test` | Always |
| `integration_cpu` | Real FAISS-CPU + embeddings | `make test-faiss-cpu` | On main |
| `integration_gpu` | Real FAISS-GPU + CUDA | `make test-faiss-gpu` | Manual only |
| `e2e` | Full system, all parsers | `make test-e2e` | Manual |
| `faiss` | Any FAISS-dependent test | `make test-faiss-cpu` | On main |

### Run Tests

```bash
make test              # Unit tests (fast) - what CI runs
make test-faiss-cpu    # FAISS CPU integration
make test-faiss-gpu    # FAISS GPU (requires GPU)
make test-e2e          # End-to-end tests
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

@pytest.mark.e2e
def test_full_pipeline():
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

**Format:** `type(scope): subject` - **scope required!**

**Types:** `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`

**Examples:**
```bash
feat(api): add SSE streaming
fix(parser): handle empty PDFs
chore(deps): update faiss-gpu
ci(workflows): improve release automation
```

❌ `feat: add feature` - missing scope
✅ `feat(api): add feature` - correct

**Breaking changes:**
```bash
feat(api)!: change response format

BREAKING CHANGE: /search now returns SearchResult object
```

**Why this matters:**
- PR title check enforces this format
- Release Please uses it to generate changelogs
- Determines semantic version bumps (feat = minor, fix = patch)

---

## Pull Request Process

### Before Creating PR

```bash
make check  # Must pass
```

### PR Requirements

- ✅ Title follows `type(scope): subject` format
- ✅ CI passes (Quality & Unit Tests)
- ✅ Description includes:
    - What changed and why
    - Link to related issues
    - Breaking changes (if any)
- ✅ Tests added for new features
- ✅ Documentation updated

### Review Process

1. CI runs automatically on PR
2. Quality & Unit Tests must pass
3. At least one approval required
4. Address review comments
5. Merge to main
6. Integration tests run automatically on main
7. Release Please creates release PR if needed

### CI Failures

**If Quality & Unit Tests fail:**
```bash
# Reproduce locally
make check

# Fix issues
make fmt    # Auto-fix formatting
make lint   # Check remaining issues
make type   # Fix type errors
make test   # Fix failing tests
```

**If Integration Tests fail (after merge to main):**
```bash
# Reproduce locally
make test-faiss-cpu

# Or trigger manually to test fix
# GitHub → Actions → "Integration Tests (FAISS CPU)" → Run workflow
```

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
6. CI will validate on PR

### Add Document Parser

1. Create `src/ai_obsidian_service/parsers/parser_format.py`
2. Implement `DocumentParser` interface
3. Register in `di_selector.py`
4. Add tests
5. Run `make check`

### Update Dependencies

```bash
# Edit environment file
vi environment.gpu.yml

# Recreate environment
make env-remove-gpu
make env-create-gpu

# Update CI workflow if needed (usually not required)
```

**Note:** CI uses `environment.yml` automatically. No workflow changes needed for dependency updates.

### Debug CI Failures

1. **Check GitHub Actions logs** for error details
2. **Reproduce locally:**
   ```bash
   # For Quality & Unit Tests failures
   make check

   # For Integration Tests failures
   make test-faiss-cpu

   # For E2E Tests failures
   make test-e2e
   ```
3. **Verify environment:**
   ```bash
   make env-validate-cpu
   ```
4. **Check conda environment matches CI:**
   ```bash
   conda activate aiobs-cpu
   python -c "import numpy; print(numpy.__version__)"
   python -c "import fastapi; print(fastapi.__version__)"
   ```

---

## Release Process

**Automated via Release Please:**

1. Commit with `feat:` or `fix:` to main
2. Release PR created automatically (version + CHANGELOG)
3. Review and merge Release PR
4. Git tag + GitHub Release created
5. Docker images published to ghcr.io

**Pull images:**
```bash
docker pull ghcr.io/USERNAME/ai-obsidian-service:latest-cpu
docker pull ghcr.io/USERNAME/ai-obsidian-service:latest-gpu
```

**Tags:** `VERSION-cpu`, `VERSION-gpu`, `latest-cpu`, `latest-gpu`

### Manual Release Testing

```bash
# Before merging release PR, test locally
make test-all

# Or trigger full CI manually
gh workflow run integration-tests-cpu.yml
gh workflow run e2e-tests.yml
```

---

## Getting Help

- Check logs for error messages
- Review existing issues/PRs
- Ask in GitHub Discussions
- Check documentation: http://127.0.0.1:8000/docs
- Review CI workflow files in `.github/workflows/`

---

## Code of Conduct

Follow our [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful and professional.
