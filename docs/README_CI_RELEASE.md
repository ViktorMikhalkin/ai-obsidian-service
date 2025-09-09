# CI/CD with release-please (solo-friendly)

This package includes:
- **ci.yml** — test every PR (pytest)
- **deploy-dev.yml** — mark dev deployment on non-draft PRs
- **deploy-prod.yml** — mark prod deployment on **tags `v*`**, with manual approval
- **release-please.yml** — auto-open a Release PR on pushes to `main`, generate CHANGELOG & version; merging that PR creates a tag `vX.Y.Z`

## How it flows
1. Open a PR → `ci` runs tests, `deploy-dev` sets an Environments: dev status on the PR.
2. Merge your PR to `main` → `release-please` updates/opens a Release PR with CHANGELOG and version.
3. Merge the Release PR → a tag `vX.Y.Z` is created.
4. Tag push triggers `deploy-prod` → click **Approve** in Actions to set Environments: prod.

> There is no real deployment yet — only statuses. Replace the placeholder URL steps with your real SSH/compose/k8s steps later.

## Setup
- Copy `.github/workflows/*.yml` and the two files at repo root:
  - `.release-please-manifest.json`
  - `release-please-config.json`
- In **Settings → Branches**, protect `main` and set **required check**: `ci`.
- In **Settings → Environments**, optionally create `dev` and `prod`.
- Use **Conventional Commits** (`feat:`, `fix:`, `docs:`…), so release-please can bump versions correctly.
