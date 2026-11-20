# Publishing guide

This project ships two packages from the same monorepo, each isolated in its own
directory:

| Package | Registry | Directory | Release tag |
| --- | --- | --- | --- |
| `ort-vision-sdk` | PyPI | `sdk-python/` | `v<MAJOR.MINOR.PATCH>` |
| `@mauriciobenjamin700/ort-vision-sdk-web` | npm | `sdk-js-web/` | `web-v<MAJOR.MINOR.PATCH>` |

**Everything is published from your machine**, through `Makefile` targets: the
packages by
[`scripts/publish.sh`](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/scripts/publish.sh),
the site by `mkdocs gh-deploy`. This repository does not use GitHub Actions — the
workflows under `.github/workflows/` are disabled and kept only as history.
Pushing a tag **publishes nothing**.

There are two stages: **credentials** (once per machine) and **release** (every
time a new version ships).

!!! note
    This is the English mirror of the Portuguese [Publicação](publishing.md)
    page, which is the source of truth.

## Prerequisites

- `git`, [`uv`](https://docs.astral.sh/uv/), `node >= 20` with `npm`, and an
  authenticated [`gh`](https://cli.github.com).
- An account with 2FA on [pypi.org](https://pypi.org) and
  [npmjs.com](https://www.npmjs.com), allowed to publish both packages.

## Credentials

No credential enters the repository. `publish.sh` checks both **before** creating
any branch or tag, and stops with a message naming what is missing.

- **PyPI:** create a token scoped to the `ort-vision-sdk` project only at
  <https://pypi.org/manage/account/token/> and export it in the shell that runs
  the release: `export UV_PUBLISH_TOKEN="pypi-..."`. `uv publish` reads it.
- **npm:** `npm login`, then `npm whoami` must answer your user.

!!! warning "Trusted Publishing and provenance are gone"
    OIDC (Trusted Publishing on PyPI, `--provenance` on npm) only exists inside
    a CI runner. Publishing locally authenticates with a token/session, and the
    npm package page no longer shows a provenance attestation.

## Release flow with `make`

The [Makefile](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/Makefile)
delegates to
[scripts/release.sh](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/scripts/release.sh):
it creates a release branch, bumps the version, validates, commits, tags,
**publishes**, pushes branch and tag, and opens a PR to `main`.

```bash
make help                                # every target
make releases                            # tag history per project

make release PROJECT=python TAG=0.3.0    # full release
make release PROJECT=web    TAG=0.3.0
make publish PROJECT=python TAG=0.3.0    # publish a tag that already exists locally

make docs-build                          # build the site, publish nothing
make docs-deploy                         # build and push the gh-pages branch
```

`TAG` is just the version number (e.g. `0.3.0`); the prefix (`v` for Python,
`web-v` for npm) is added for you.

### Steps for a release

1. **Date the CHANGELOG** — move `## [Unreleased]` to `## [0.3.0] - YYYY-MM-DD`,
   add the link at the bottom, and commit to `main` (the script requires a clean
   working tree).
2. **Rehearse the publish** — no credential, no tag, nothing leaves the machine:
   `REHEARSAL=1 scripts/publish.sh python <current version>`. It runs the same
   guards, build and smoke-install as the real publish and stops before the
   upload.
3. (Optional) **Dry-run the release**: `make release PROJECT=python TAG=0.3.0 DRY_RUN=1`
   creates the branch, bump and tag without publishing, pushing or opening a PR.
4. **Run the release**: `make release PROJECT=python TAG=0.3.0`. In order it
   checks the tag, warns if the CHANGELOG lacks `[0.3.0]`, **checks the publish
   credential**, creates the branch, bumps, validates
   ([`scripts/validate.sh`](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/scripts/validate.sh)),
   commits, tags, regenerates `RELEASES.md`, **publishes**, pushes and opens the
   PR.
5. **Merge the PR** — it brings the bump and `RELEASES.md` to `main`. The version
   is already published by then.

!!! danger "Publishing is irreversible"
    Neither PyPI nor npm lets a version be published twice. If the publish fails
    **before** the upload, branch and tag stay local and the script prints how
    to discard them. After the upload, the way back is yank/deprecate and a new
    PATCH.

### Accepted variables

| Variable | Description |
| --- | --- |
| `PROJECT=python\|web` | Which SDK (required for `make release` and `make publish`). |
| `TAG=0.3.0` | Version without prefix, semver format. |
| `DRY_RUN=1` | Local branch + commit + tag; skip publish, push and PR. |
| `SKIP_VALIDATE=1` | Skip lint/typecheck/build (only if just validated). |
| `BASE_BRANCH=...` | Target branch of the PR (default `main`). |
| `REHEARSAL=1` | `publish.sh` only: guards, build and smoke-install, no upload. |

### What `publish.sh` does

The same guards as the old workflows, in the same order, each there because of a
defect that already happened:

1. **Checks version against tag**, before anything is uploaded.
2. **Validates the artifact**: build, `twine check` (Python) or typecheck + tests
   (web), and a smoke-install in a clean venv/project that imports the tasks,
   the result envelopes and the graph helpers. Python also checks the wheel is
   not empty.
3. **Publishes** with `uv publish` or `npm publish --access public`.
4. **Reads back from the registry**: PyPI through the simple index, npm through
   `npm view --prefer-online` (version and the `latest` dist-tag), each for up
   to five minutes — both caches lie right after an upload.
5. **Creates or updates the GitHub Release** for the tag, with the CHANGELOG
   section as notes and the artifact attached. A `gh` API call, not Actions.

!!! tip "Reconciling what was left behind"
    `make releases-sync-dry`, `make releases-sync` and `make releases-check`
    (tags × published versions × Releases, exits non-zero on a mismatch). Tags
    deliberately never published are recorded in
    [`scripts/never-published.tsv`](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/scripts/never-published.tsv).

### When something goes wrong

- **`UV_PUBLISH_TOKEN vazio` / `npm sem sessão`** → see Credentials. Nothing was
  created.
- **Validation failed midway** → `git checkout -- sdk-python/` (or `sdk-js-web/`),
  fix, run again.
- **Publish failed before the upload** → branch and tag are local; fix and run
  `make publish PROJECT=... TAG=...`.
- **Read-back failed but the version is on the registry** → the publish worked
  and only propagation lagged. Do **not** run `make publish` again (the upload
  would be refused as a duplicate): push branch and tag by hand, create the
  GitHub Release with `make releases-sync`, and open the PR.

## Docs site

The bilingual site is published to GitHub Pages from the `gh-pages` branch:
`make docs-build` builds it with `--strict`, `make docs-deploy` builds and pushes.
Both run in an ephemeral `uv` environment built from `docs/requirements.txt`. A
docs-only change gets no version: a `docs:` commit on `main` and
`make docs-deploy`.

## Versioning

Both packages follow [SemVer](https://semver.org). Keep them in **lockstep** when
a change affects both (e.g. a new public type), independent otherwise. The version
lives in two files per package, kept in sync by `release.sh`:

- Python: `sdk-python/pyproject.toml` (`project.version`) and
  `sdk-python/src/ort_vision_sdk/__init__.py` (`__version__`).
- Web: `sdk-js-web/package.json` (`version`) and `sdk-js-web/src/index.ts`
  (`VERSION`).

## Rollback / yank

- **PyPI:** a published version cannot be deleted — **yank** it at
  <https://pypi.org/manage/project/ort-vision-sdk/releases/> and publish a fixed
  patch right away.
- **npm:** `npm unpublish` within 72 h, `npm deprecate` afterwards. Never reuse
  an unpublished version number.

## Common problems

- **`twine check` complains about the README** → PyPI uses strict CommonMark;
  use absolute URLs for images/links.
- **`uv publish` answers `403 Forbidden`** → the token is not scoped to
  `ort-vision-sdk`, expired or was revoked.
- **`npm publish` answers `E404` on the `PUT`** → no valid session; for a scoped
  package the registry answers 404 instead of 401. `npm whoami`, then
  `npm login`.
- **Duplicate version** → both registries refuse a re-upload; bump PATCH.
- **The site did not change after the merge** → merging does not publish it; run
  `make docs-deploy`.
