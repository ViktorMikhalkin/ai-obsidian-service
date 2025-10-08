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

### Workflow Architecture

Our CI uses a **layered testing strategy** for fast feedback and comprehensive coverage:

```
┌─────────────────────────────────────────────────┐
│ ci.yml - FAST FEEDBACK (every PR/push)         │
│ • Unit tests only                               │
│ • Memory backend, no external deps              │
│ • Runs: make lint, make type, make test         │
│ • Duration: ~2-3 minutes                        │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ faiss-cpu.yml - INTEGRATION (main + manual)     │
│ • Real FAISS, embeddings, ML stack              │
│ • Runs: make test-faiss-cpu                     │
│ • Duration: ~5-10 minutes                       │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ e2e.yml - FULL SYSTEM (manual only)             │
│ • All parsers (PDF, EPUB, etc.)                 │
│ • Runs: make test-e2e                           │
│ • Duration: ~10-20 minutes                      │
└─────────────────────────────────────────────────┘
```

### CI Workflows

| Workflow | Trigger | Purpose | Makefile Equivalent |
|----------|---------|---------|---------------------|
| **ci.yml** | Every push/PR | Fast quality gate | `make check` |
| **faiss-cpu.yml** | Main branch + manual | Integration testing | `make test-faiss-cpu` |
| **e2e.yml** | Manual only | System testing | `make test-e2e` |
| **pr-title-check.yml** | PR events | Enforce conventions | N/A |
| **release-please.yml** | Main push | Auto releases | N/A |

### Local CI Simulation

**Replicate CI locally before pushing:**

```bash
# Run what quality-and-unit-tests.yml runs
make lint
make type
make test

# Or all at once
make check

# Run integration tests (like integration-tests-cpu.yml)
make test-faiss-cpu

# Run e2e tests (like e2e-tests.yml)
make test-e2e
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
# Actions tab → "CI (integration-cpu, py311)" → Run workflow

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
| `unit` (default) | Fast tests, mocks/fakes | `make test` | Always (ci.yml) |
| `integration_cpu` | Real FAISS-CPU + embeddings | `make test-faiss-cpu` | On main (faiss-cpu.yml) |
| `integration_gpu` | Real FAISS-GPU + CUDA | `make test-faiss-gpu` | Manual only |
| `e2e` | Full system, all parsers | `make test-e2e` | Manual (e2e.yml) |
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
ci(workflows): migrate to conda-based CI
```

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
make check  # Must pass (same as quality-and-unit-tests.yml)
```

### PR Requirements

- ✅ Title follows Conventional Commits (checked by pr-title-check.yml)
- ✅ CI passes (ci.yml must be green)
- ✅ Description includes:
    - What changed and why
    - Link to related issues
    - Breaking changes (if any)
- ✅ Tests added for new features
- ✅ Documentation updated

### Review Process

1. CI runs automatically on PR
2. ci.yml must pass (unit tests + quality checks)
3. At least one approval required
4. Address review comments
5. Merge to main
6. Integration tests run automatically on main (faiss-cpu.yml)
7. Release Please creates release PR if needed

### CI Failures

**If ci.yml fails:**
```bash
# Reproduce locally
make check

# Fix issues
make fmt    # Auto-fix formatting
make lint   # Check remaining issues
make type   # Fix type errors
make test   # Fix failing tests
```

**If faiss-cpu.yml fails (after merge to main):**
```bash
# Reproduce locally
make test-faiss-cpu

# Or trigger manually to test fix
# GitHub → Actions → "CI (integration-cpu, py311)" → Run workflow
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
   # For quality-and-unit-tests.yml failures
   make check

   # For integration-tests-cpu.yml failures
   make test-faiss-cpu

   # For e2e-tests.yml failures
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

### Automated Releases

We use [Release Please](https://github.com/googleapis/release-please) for automated releases:

1. **Commit with conventional commits** to main
2. **Release Please bot** analyzes commits
3. **Release PR is created** with version bump and changelog
4. **Review and merge** the release PR
5. **GitHub release is created** automatically
6. **Package is published** (if configured)

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
