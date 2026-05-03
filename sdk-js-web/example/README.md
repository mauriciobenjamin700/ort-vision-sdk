# Browser demo

Single-page demo that runs all three SDK tasks in the browser — image
classification, object detection and instance segmentation — using
`@mauriciobenjamin700/ort-vision-sdk-web` and `onnxruntime-web`. Bring your
own `.onnx` model or paste a URL (a Hugging Face preset is filled in by
default for each task).

## Run

From the parent `sdk-js-web/` directory:

```bash
npm install
npm run build              # produces dist/index.js, which the demo imports
```

Then serve `sdk-js-web/` with any static HTTP server:

```bash
python3 -m http.server 8000
```

Open <http://localhost:8000/example/>.

VS Code's **Live Server** extension also works — right-click
`example/index.html` → "Open with Live Server" (default port 5500). The page
serves the same way as `python3 -m http.server`.

## Notes

- The page loads `onnxruntime-web` from the jsDelivr CDN
  (`onnxruntime-web@1.17`). The matching `.wasm` files are fetched from the
  same CDN via `ort.env.wasm.wasmPaths`, so no local install of WASM
  artifacts is required.
- WebGPU is preferred when available (Chromium-based browsers); ORT-Web
  falls back to WASM transparently when WebGPU is missing.
- For very large models (>50 MB) over a slow connection, prefer the local
  file picker — the URL fetch reads the whole file before instantiating the
  session.
- Default model presets:
  - **Classification:** `Xenova/mobilenetv2_100` (224×224, ImageNet)
  - **Detection:** `Xenova/yolov10n` (anchor-free YOLO, COCO)
  - **Segmentation:** `Xenova/yolov8n-seg` (YOLO seg, COCO)
- Classification needs labels. The demo defaults to fetching the 1000
  ImageNet labels from a public GitHub raw URL; you can also upload your
  own `.txt` (one per line) or `.json` file, or pick "None" to see raw
  class IDs (`class_0`, `class_1`, …).

## Troubleshooting

If the page loads but nothing happens when you click **Load model + run**:

1. Open browser devtools → Console.
2. **`Failed to fetch ../dist/index.js`** → run `npm run build` from
   `sdk-js-web/`. The demo imports the local build output.
3. **`CORS error` on the model URL** → switch to a local file or use a
   CORS-friendly host. Hugging Face, jsDelivr and the ONNX Model Zoo all
   work; many random servers don't allow `fetch()` from a different origin.
4. **Module specifier `ort-vision-sdk-web` not found** → the importmap is
   not being applied. Make sure you're serving over HTTP (not opening the
   file directly with `file://`); the demo requires an HTTP origin.
