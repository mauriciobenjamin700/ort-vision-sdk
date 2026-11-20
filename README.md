# ort-vision-sdk

> **Documentação / Docs:** site bilíngue no GitHub Pages —
> **[Português (BR)](https://mauriciobenjamin700.github.io/ort-vision-sdk/)** ·
> **[English (US)](https://mauriciobenjamin700.github.io/ort-vision-sdk/en/)**.
> Use o seletor PT-BR / EN-US no topo do site para alternar o idioma.

High-level SDKs for computer vision inference on top of [ONNX Runtime](https://onnxruntime.ai/).

The repo distributes two sibling packages — same task-oriented API (`Classifier`, `Detector`, `Segmenter`, `DetectClassify`), same typed result shapes (`BoundingBox`, `ClassificationResult`, `DetectionResult`) — one for Python servers/scripts and one for the browser.

Models can also be **fused**: `ort_vision_sdk.compose` rewrites a detector and a classifier into a single `.onnx`, so what would be two sessions and a per-crop round trip becomes one file, one session, and no round trip. The fused file runs unchanged in Python and in the browser — see [Fused pipelines](https://mauriciobenjamin700.github.io/ort-vision-sdk/en/guia/pipeline/).

| Package               | Registry | Directory                          | Install                             |
| --------------------- | -------- | ---------------------------------- | ----------------------------------- |
| `ort-vision-sdk`      | PyPI     | [`sdk-python/`](sdk-python)        | `pip install ort-vision-sdk`        |
| `@mauriciobenjamin700/ort-vision-sdk-web` | npm      | [`sdk-js-web/`](sdk-js-web)        | `npm install @mauriciobenjamin700/ort-vision-sdk-web onnxruntime-web` |

Each package is self-contained: its own `README.md`, `LICENSE`, `CHANGELOG.md` and build tooling. Open the directory you care about for usage details.

## Releasing

Releases are published **from your machine**, not by GitHub Actions (the
workflows are disabled; pushing a tag publishes nothing). See
[docs/publishing.md](docs/publishing.md) for credentials and the full
step-by-step. Short version:

```bash
export UV_PUBLISH_TOKEN="pypi-..."      # Python; npm needs `npm login`
make release PROJECT=python TAG=x.y.z   # bump, validate, tag, publish, push, PR
make release PROJECT=web    TAG=x.y.z
```

## Documentation

The full bilingual documentation lives on GitHub Pages:

- Português (BR): <https://mauriciobenjamin700.github.io/ort-vision-sdk/>
- English (US): <https://mauriciobenjamin700.github.io/ort-vision-sdk/en/>

The site is built with MkDocs Material + `mkdocs-static-i18n` and deployed with
`make docs-deploy` (`mkdocs gh-deploy` to the `gh-pages` branch, run locally).
Sources live under [`docs/`](docs) (PT-BR base files; EN translations as
`*.en.md`).

> Local preview (optional): `pip install -r docs/requirements.txt && mkdocs serve`.

## Status

Alpha — APIs may change between minor versions until 1.0.

## License

[MIT](LICENSE).
