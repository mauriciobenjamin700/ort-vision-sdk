# ort-vision-sdk

> **Documentação / Docs:** site bilíngue no GitHub Pages —
> **[Português (BR)](https://mauriciobenjamin700.github.io/ort-vision-sdk/)** ·
> **[English (US)](https://mauriciobenjamin700.github.io/ort-vision-sdk/en/)**.
> Use o seletor PT-BR / EN-US no topo do site para alternar o idioma.

High-level SDKs for computer vision inference on top of [ONNX Runtime](https://onnxruntime.ai/).

The repo distributes two sibling packages — same task-oriented API (`Classifier`, `Detector`), same typed result shapes (`BoundingBox`, `ClassificationResult`, `DetectionResult`) — one for Python servers/scripts and one for the browser.

| Package               | Registry | Directory                          | Install                             |
| --------------------- | -------- | ---------------------------------- | ----------------------------------- |
| `ort-vision-sdk`      | PyPI     | [`sdk-python/`](sdk-python)        | `pip install ort-vision-sdk`        |
| `@ort-vision-sdk/web` | npm      | [`sdk-js-web/`](sdk-js-web)        | `npm install @ort-vision-sdk/web onnxruntime-web` |

Each package is self-contained: its own `README.md`, `LICENSE`, `CHANGELOG.md` and build tooling. Open the directory you care about for usage details.

## Releasing

See [docs/publishing.md](docs/publishing.md) for the full step-by-step (Trusted Publishing on PyPI, npm provenance, tag conventions). Short version:

- Python release: bump version in `sdk-python/`, tag `v<x.y.z>`, push.
- Web release: bump version in `sdk-js-web/`, tag `web-v<x.y.z>`, push.

GitHub Actions ([.github/workflows](.github/workflows)) handles the rest.

## Documentation

The full bilingual documentation lives on GitHub Pages:

- Português (BR): <https://mauriciobenjamin700.github.io/ort-vision-sdk/>
- English (US): <https://mauriciobenjamin700.github.io/ort-vision-sdk/en/>

The site is built with MkDocs Material + `mkdocs-static-i18n` and deployed by the
[`Docs`](.github/workflows/docs.yml) GitHub Actions workflow on every push to
`main` that touches `docs/`, `mkdocs.yml`, or the workflow itself. Sources live
under [`docs/`](docs) (PT-BR base files; EN translations as `*.en.md`).

> Local preview (optional): `pip install -r docs/requirements.txt && mkdocs serve`.

## Status

Alpha — APIs may change between minor versions until 1.0.

## License

[MIT](LICENSE).
