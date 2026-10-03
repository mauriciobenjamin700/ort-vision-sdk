/**
 * Make Node a runtime the SDK can execute in, for the end-to-end tests.
 *
 * Two gaps, both Node-only:
 *
 * - **No canvas.** See below.
 * - **No multi-threaded WASM.** `onnxruntime-web` warns that it does not support
 *   threads under Node, and up to 1.20 its threaded build fails to instantiate
 *   there (`memory import must be a WebAssembly.Memory object`), so no session
 *   could be created on the peer range's floor. One thread is what Node runs
 *   anyway; the browser keeps ORT's default.
 *
 * Give Node the canvas surface the preprocessing pipelines draw on.
 *
 * `LetterboxPipeline` resizes through a 2D canvas, and Node has none, so the
 * detector and the segmenter cannot run a frame here without one. Every other
 * suite sidesteps that by mocking `onnxruntime-web` and testing the maths alone,
 * which is how three defects shipped: each lived in the seam between a real ORT
 * session and the task reading it (FP16 feeds, #50; the 1 MB graph read, #54;
 * mask coefficients counted as classes once ORT reported output shapes).
 *
 * `@napi-rs/canvas` is a real Skia canvas, so the letterbox resize and the
 * inverse geometry run as they would in a browser. It is a dev dependency only:
 * nothing in the published package imports it.
 *
 * Import this module before anything from `src/`, so the globals exist when
 * the pipelines first allocate a canvas.
 */
import { createCanvas, ImageData as NapiImageData } from "@napi-rs/canvas";
import * as ort from "onnxruntime-web";

/**
 * Stand-in for `OffscreenCanvas` that returns a napi canvas.
 *
 * The SDK only calls `new OffscreenCanvas(width, height)` and the 2D context
 * methods on the result, all of which the napi canvas implements.
 */
class OffscreenCanvasShim {
  /**
   * @param width Canvas width in pixels.
   * @param height Canvas height in pixels.
   */
  constructor(width: number, height: number) {
    return createCanvas(width, height) as unknown as OffscreenCanvasShim;
  }
}

const globals = globalThis as Record<string, unknown>;
if (globals["OffscreenCanvas"] === undefined) globals["OffscreenCanvas"] = OffscreenCanvasShim;
if (globals["ImageData"] === undefined) globals["ImageData"] = NapiImageData;

ort.env.wasm.numThreads = 1;
