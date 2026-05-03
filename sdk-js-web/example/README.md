# Browser demo

Single-page demo that runs YOLO detection / instance segmentation in the
browser using `@ort-vision-sdk/web` and `onnxruntime-web`. Bring your own
`.onnx` model or paste a URL (a Hugging Face YOLOv10n preset is filled in by
default).

## Run

From the parent `sdk-js-web/` directory:

```bash
npm install
npm run build              # produces dist/index.js, which the demo imports
python3 -m http.server 8000   # any static server works
```

Then open <http://localhost:8000/example/>.

## Notes

- The page loads `onnxruntime-web` from the jsDelivr CDN
  (`onnxruntime-web@1.17`). The matching `.wasm` files are loaded from the
  same CDN via `ort.env.wasm.wasmPaths`, so no local install of WASM
  artifacts is required.
- WebGPU is preferred when available (Chromium-based browsers); ORT-Web
  falls back to WASM transparently when WebGPU is missing.
- For very large models (>50 MB) over a slow connection, prefer the local
  file picker — the URL fetch reads the whole file before instantiating the
  session.
