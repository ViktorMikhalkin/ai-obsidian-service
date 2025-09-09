# Contributing Guide

Thank you for your interest in contributing to **AI↔Obsidian Service**!
This guide explains how to work with the repository and submit changes.

## 💡 How to contribute

1. Fork the repository.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
3. Make changes and add tests when applicable.
4. Run checks locally:
   ```bash
   pytest -v
   python scripts/export_openapi.py
   ```
5. Open a Pull Request to `main`.

## 📝 Commit messages

We use **Conventional Commits** validated by husky + commitlint.

Format:
```
<type>(<scope>): <message>
```
Examples:
- `feat(api): add /index/stats endpoint`
- `fix(parser): handle empty PDF metadata`
- `docs(readme): update install instructions`

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`  
Allowed scopes: `api`, `index`, `parser`, `faiss`, `rag`, `cli`, `docs`, `ci`, `repo`

## 🔄 Pull Requests

- All PRs must pass CI (pytest + OpenAPI export).
- Auto-merge is enabled: once checks are green, the PR can merge automatically.
- Use the PR template (`.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`).

## 🐛 Issues

Use the provided templates for:
- 🐛 Bug report
- ✨ Feature request
- 🧩 Task

## 📦 Releases

We use **release-please**:
- Conventional Commits drive version bumps and `CHANGELOG.md`.
- Annotated tags are created automatically.
- GitHub Release notes are generated from commit messages.

## 🛠 Local development

### Environment
```bash
conda env create -n ai-obsidian -f environment.yml
conda activate ai-obsidian
```

### Run service
```bash
uvicorn indexer.app:app --host 127.0.0.1 --port 8000 --reload
```

### Run tests
```bash
pytest -v
```

### Export OpenAPI
```bash
python scripts/export_openapi.py
```

Swagger UI: http://127.0.0.1:8000/docs  
ReDoc: http://127.0.0.1:8000/redoc

## 🤝 Code of Conduct

Participation in this project implies agreement with the [Contributor Covenant](https://www.contributor-covenant.org/).
