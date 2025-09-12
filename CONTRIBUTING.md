# Contributing


## Conventional Commit policy
Types: feat, fix, docs, style, refactor, test, chore, ci, build, **perf**.
Scopes are configured in `pyproject.toml` under `[tool.ccpolicy]`.


### Allowed scopes
Используйте один из разрешённых scopes (редактируется в `pyproject.toml`):
```
api, auth, cli, infra, deps, docs, tests, ci, index, parser, faiss, rag, repo, main
```
PR заголовки от Release Please используют `chore(main): release X.Y.Z`, поэтому `main` включён в список.


## Conventional Commits — strict scope
- **Scope обязателен** и должен быть одним из: `index`, `parser`, `faiss`, `rag`, `repo`, `main`.
- Примеры:
  - `feat(index): add posting lists compaction`
  - `perf(faiss): speed up IVF search`
  - `fix(parser): handle empty inputs`
  - `chore(repo): bump hooks`
  - `refactor(rag)!: remove legacy retriever`


## Pre-commit hooks

- Project includes pre-commit configuration. To enable:
  ```bash
  pre-commit install
  pre-commit run --all-files
  ```
- Hooks will run automatically on each commit.

## Debugging in IDE (PyCharm/IntelliJ, VS Code)

- Environments are created from conda YAMLs:
    - CPU: `make setup-cpu` (environment.yml)
    - GPU: `make ENV=aiobs-gpu setup-gpu` (environment.gpu.yml)
- In IDE, select the existing conda interpreter created by the Makefile (do not create a new venv).
- Recommended Run/Debug configurations:
    - API (dev): run `uvicorn` with module `indexer.app:app`, `--reload`, working directory = project root.
    - CLI: run `python -m cli.aiobs ...`.
    - Tests: `pytest tests/`.
- Environment variables:
    - Use `.env` or `.env.local` (see `.env.example`).
    - Common: `AIOBS_TEST_MODE=1`, optional `AIOBS_TEST_INDEX_DIR=/tmp/aiobs-test-index`, `OLLAMA_MODEL=llama3.1:8b`.
- For GPU: duplicate Run/Debug configs and switch interpreter to the GPU conda env.
