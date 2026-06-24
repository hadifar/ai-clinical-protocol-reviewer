# Contributing

Thanks for working on AI Clinical Protocol Reviewer. This document describes how
we branch, commit, and ship changes. The rules here aren't just style — commit
messages drive automatic versioning, so following them keeps releases correct.

## Getting started

```bash
uv sync --dev        # install runtime + dev tools (ruff, pyright, pytest, pre-commit, commitizen)
pre-commit install   # wire up the git hooks (runs ruff, pyright, and commit-msg linting locally)
```

After `pre-commit install`, the hooks run automatically on every commit. They
format and lint your changes, type-check, and reject commit messages that don't
follow the convention below — so most CI failures are caught before you push.

## Branching

We use a trunk-based flow: `main` is the single long-lived branch and is always
releasable. Do your work on a short-lived branch off `main`:

```bash
git switch -c feat/short-description main
# ...work...
git push -u origin feat/short-description
```

Name branches by type: `feat/`, `fix/`, `docs/`, `chore/`, etc. Delete the branch
after it merges (GitHub does this for you on merge).

## Commits

We follow [Conventional Commits](https://www.conventionalcommits.org/). This is
**enforced** by a `commit-msg` hook and is what [commitizen](https://commitizen-tools.github.io/commitizen/)
uses to compute the next version on `main`.

Format: `type(optional scope): description`

| Type | Use for | Version effect (on `main`) |
|------|---------|----------------------------|
| `feat` | a new feature | **minor** bump (1.1.0 → 1.2.0) |
| `fix` | a bug fix | **patch** bump (1.1.0 → 1.1.1) |
| `docs`, `chore`, `refactor`, `test`, `ci`, `style`, `perf` | everything else | no bump |
| any type + `BREAKING CHANGE:` in body | incompatible change | **major** bump (1.1.0 → 2.0.0) |

Examples:

```
feat(ingestion): index sparse BM25 embeddings alongside dense
fix(search): handle empty query without raising
docs: document the IE agent tool-calling requirement
```

## Before you push

The CI `lint` job runs exactly these — run them locally to avoid a red build:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest          # if the change touches behavior covered by tests
```

## Pull requests

1. Open a PR against `main`:
   ```bash
   gh pr create --fill
   ```
2. CI (`lint`) must pass. `main` is protected by a ruleset that requires a
   passing PR before merge.
3. **Squash-merge**, and make sure the squash commit **title is a valid
   Conventional Commit** — that title (not the individual commits) is what
   versioning reads. A wrong title means a wrong or missing version bump.

## Releases

Releases are automatic. When a commit lands on `main`, the **Bump version**
workflow runs commitizen, which:

- computes the new version from the commit messages since the last tag,
- updates `version` in `pyproject.toml`,
- updates the changelog,
- commits the change as `bump: version X → Y` and creates a `vX.Y.Z` tag.

You don't bump versions or edit the changelog by hand.
