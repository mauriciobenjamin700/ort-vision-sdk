# Publishing guide

This project ships two packages from the same monorepo, each isolated in its own
directory:

| Package | Registry | Directory | Release tag |
| --- | --- | --- | --- |
| `ort-vision-sdk` | PyPI | `sdk-python/` | `v<MAJOR.MINOR.PATCH>` |
| `@mauriciobenjamin700/ort-vision-sdk-web` | npm | `sdk-js-web/` | `web-v<MAJOR.MINOR.PATCH>` |

The release flows are automated in
[.github/workflows/release-pypi.yml](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/.github/workflows/release-pypi.yml)
and
[.github/workflows/release-npm.yml](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/.github/workflows/release-npm.yml).
You publish by pushing a tag — GitHub Actions does the rest.

There are two stages: **initial setup** (once per package) and **release** (every
time a new version ships).

!!! note
    This is the English mirror of the Portuguese
    [Publicação](publishing.md) page, which is the source of truth and carries
    the full step-by-step (Trusted Publishing on PyPI, npm provenance, the
    `make release` flow, rollback/yank, and troubleshooting).

## Prerequisites

- GitHub repository: `https://github.com/mauriciobenjamin700/ort-vision-sdk`.
- An account with 2FA on [pypi.org](https://pypi.org) and
  [npmjs.com](https://www.npmjs.com).
- Locally: `python >= 3.10`, `node >= 18`, `git`.

## PyPI — Trusted Publishing

Trusted Publishing uses OIDC: GitHub Actions authenticates directly to PyPI with
no stored token. Add a **pending publisher** at
<https://pypi.org/manage/account/publishing/> with the project name
`ort-vision-sdk`, owner `mauriciobenjamin700`, repository `ort-vision-sdk`,
workflow `release-pypi.yml`, environment `pypi`. Validate first on
[test.pypi.org](https://test.pypi.org). Create the `pypi` environment under
**Settings → Environments** (optionally with required reviewers) — no secret is
needed, OIDC handles authentication.

## npm

The `package.json` uses `@mauriciobenjamin700/ort-vision-sdk-web` (a personal
scope). Generate a granular automation token with read/write on the package and
store it as the `NPM_TOKEN` repository secret. The workflow passes
`--provenance`, which requires a public repo with OIDC and a correct
`repository.url`.

## Release flow with `make` (recommended)

The [Makefile](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/Makefile)
(delegating to
[scripts/release.sh](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/scripts/release.sh))
automates the whole flow: it creates a dedicated release branch, bumps the
version, validates locally, commits, tags, pushes the branch + tag, and opens a
PR to `main`. `main` never gets a direct push.

```bash
make help                                # list every target
make releases                            # tag history per project

make release PROJECT=python TAG=0.3.0    # Python release
make release PROJECT=web    TAG=0.3.0    # Web release
make release-python TAG=0.3.0            # shortcut
make release-web    TAG=0.3.0            # shortcut
```

`TAG` is just the version number (e.g. `0.3.0`); the Makefile adds the right
prefix (`v` for Python, `web-v` for npm). Use `DRY_RUN=1` to do everything
locally without pushing or opening a PR.

### Steps for a release

1. **Update the CHANGELOG** — move `## [Unreleased]` to
   `## [0.3.0] - YYYY-MM-DD` in the package's `CHANGELOG.md` (the only manual
   step).
2. Commit the changelog (the Makefile requires a clean working tree).
3. (Recommended) Run a dry-run: `make release PROJECT=python TAG=0.3.0 DRY_RUN=1`.
4. Run the real release: `make release PROJECT=python TAG=0.3.0`.
5. **The tag triggers the workflow** — track it under **GitHub → Actions** and
   approve the `pypi` deployment when prompted. Publishing runs independently of
   the PR merge (the tag is the source of truth).
6. Merge the PR when ready to propagate the version bump to `main`.

### Accepted variables

| Variable | Description |
| --- | --- |
| `PROJECT=python\|web` | Which SDK to release (required for `make release`). |
| `TAG=0.3.0` | Version without prefix, semver format. |
| `DRY_RUN=1` | Do everything locally (branch + commit + tag) but skip push and PR. |
| `SKIP_VALIDATE=1` | Skip lint/typecheck/build (only if just validated manually). |
| `BASE_BRANCH=...` | Target branch of the PR (default `main`). |

## Versioning

Both packages follow [SemVer](https://semver.org). Keep them in **lockstep** when
a change affects both (e.g. a new public type), independent otherwise. Version is
carried in two files per package and must stay in sync:

- Python: `sdk-python/pyproject.toml` (`project.version`) and
  `sdk-python/src/ort_vision_sdk/__init__.py` (`__version__`).
- Web: `sdk-js-web/package.json` (`version`) and `sdk-js-web/src/index.ts`
  (`VERSION`).

## Rollback / yank

- **PyPI:** you cannot delete a published version — **yank** it
  (`twine yank ort-vision-sdk==0.2.0`) and publish a fixed patch immediately.
- **npm:** `npm unpublish` within 72 h, then `npm deprecate` afterwards. Never
  reuse an unpublished version number.

## Common problems

- **`twine check` complains about the README** → PyPI uses strict CommonMark;
  use absolute URLs for images/links.
- **`invalid-publisher`** → the Trusted Publisher fields don't match the
  workflow/environment/repo.
- **`npm publish` 403 — name disputed** → the package name is taken; rename.
- **Duplicate version** → both registries refuse a re-upload; bump PATCH.
