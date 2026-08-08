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

## npm — Trusted Publishing

The published package is `@mauriciobenjamin700/ort-vision-sdk-web` (a personal
scope, which already exists since it is your username). The workflow publishes
over **OIDC**, exactly like PyPI: no token is stored in the repository.

Register the Trusted Publisher on npmjs — this is the step that is **currently
missing**, and the only reason the npm workflow fails with `E404`:

1. Go to the package page → **Settings** → **Trusted publisher** → **GitHub
   Actions**.
2. Organization or user `mauriciobenjamin700`, repository `ort-vision-sdk`,
   workflow filename `release-npm.yml`, environment empty (the job uses none).

!!! danger "Without this, publish fails with an error that reads like permissions"
    With no Trusted Publisher and no `NODE_AUTH_TOKEN`, the registry answers
    `npm error code E404` on the `PUT`. The whole build stays green up to that
    step, so it is easy to read as network trouble. It is authentication.

Registering an `NPM_TOKEN` secret instead also works, but it keeps a long-lived
credential in the repository and loses the provenance attestation. The workflow
passes `--provenance`, which with OIDC needs no extra setup: a public repo and a
correct `repository.url`, both already in place.

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
5. **The tag triggers the workflow** — track it under **GitHub → Actions**.
   Publishing runs independently of the PR merge (the tag is the source of
   truth).

    !!! danger "No approval stands between the tag and the registry"
        This repository's `pypi` environment has **no required reviewers**, so the
        publish job never pauses: pushing the tag *is* publishing. npm works the
        same way by construction. Neither registry lets a version be replaced, so
        treat `make release` as irreversible — add required reviewers to the
        `pypi` environment first if you want that brake.

6. Merge the PR when ready to propagate the version bump to `main`.

### Accepted variables

| Variable | Description |
| --- | --- |
| `PROJECT=python\|web` | Which SDK to release (required for `make release`). |
| `TAG=0.3.0` | Version without prefix, semver format. |
| `DRY_RUN=1` | Do everything locally (branch + commit + tag) but skip push and PR. |
| `SKIP_VALIDATE=1` | Skip lint/typecheck/build (only if just validated manually). |
| `BASE_BRANCH=...` | Target branch of the PR (default `main`). |

### What the workflow does on its own once the tag lands

Pushing the tag is the last manual step. From there GitHub Actions, in order:

1. **Checks version against tag.** `web-v0.4.0` only publishes if `sdk-js-web/package.json` says `0.4.0`; `v0.6.0` only publishes if both `pyproject.toml` and `__version__` say `0.6.0`. A mismatch aborts **before** the upload — PyPI never lets a version be replaced, so this guard has to run before, not after.
2. **Validates.** Typecheck, tests, build, and a smoke-install of the packed artifact in a clean project (web imports it with `onnxruntime-web`; Python installs the wheel in a fresh venv). Python also checks the wheel did not come out empty.
3. **Publishes** over OIDC — provenance on npm, Trusted Publishing on PyPI.
4. **Reads back from the registry.** Confirms the registry really serves that version and that it is `latest`. A publish that landed under a different dist-tag, or an index that never propagated, fails here instead of passing as a green release.
5. **Creates (or updates) the GitHub Release.** Notes from the matching CHANGELOG section, artifacts attached (tarball for web, sdist + wheel for Python), titled `ort-vision-sdk (Python) X.Y.Z` / `@mauriciobenjamin700/ort-vision-sdk-web X.Y.Z`. Idempotent — re-running a tag edits the existing Release instead of failing.

!!! tip "Reconciling what was left behind"
    Old releases that never got a GitHub Release can be backfilled at once:

    ```bash
    make releases-sync-dry   # list what is missing, create nothing
    make releases-sync       # create the missing Releases
    make releases-check      # tags x published versions x Releases, side by side
    ```

    `releases-check` is the quick way to spot a tag with no published package — or the other way around.

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
