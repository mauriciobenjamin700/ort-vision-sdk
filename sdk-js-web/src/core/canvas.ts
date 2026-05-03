/**
 * Canvas plumbing shared by `io/image.ts` and `preprocess/image.ts`.
 *
 * Browsers and workers expose two canvas types (`HTMLCanvasElement` and
 * `OffscreenCanvas`) with slightly different surfaces. These helpers pick the
 * right one for the current environment, return a unified 2D context, and
 * convert between RGBA `ImageData` and the SDK's canonical HWC RGB
 * `RGBImage`.
 */

import { ImageLoadError } from "./exceptions.js";
import { RGBImage } from "../types.js";

export type Canvas2D = HTMLCanvasElement | OffscreenCanvas;
export type Context2D =
  | CanvasRenderingContext2D
  | OffscreenCanvasRenderingContext2D;

/** Allocate a `Canvas2D`, preferring `OffscreenCanvas` when available. */
export function createCanvas(width: number, height: number): Canvas2D {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(width, height);
  }
  if (typeof document !== "undefined") {
    const c = document.createElement("canvas");
    c.width = width;
    c.height = height;
    return c;
  }
  throw new ImageLoadError("No canvas implementation available in this environment.");
}

/** Get a 2D context from a canvas, throwing a typed error if the browser refuses. */
export function get2DContext(canvas: Canvas2D): Context2D {
  const ctx = canvas.getContext("2d") as Context2D | null;
  if (ctx === null) {
    throw new ImageLoadError("Failed to obtain 2D rendering context.");
  }
  return ctx;
}

/** Convert RGBA `ImageData` into the canonical HWC RGB `RGBImage`. */
export function imageDataToRGB(imageData: ImageData): RGBImage {
  const { data, width, height } = imageData;
  if (data.length !== width * height * 4) {
    throw new ImageLoadError(
      `Unexpected ImageData length ${data.length} for ${width}x${height} (expected ${
        width * height * 4
      }).`,
    );
  }
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i] as number;
    rgb[j + 1] = data[i + 1] as number;
    rgb[j + 2] = data[i + 2] as number;
  }
  return new RGBImage(rgb, width, height);
}

/** Convert an `RGBImage` into RGBA `ImageData` (alpha forced to 255). */
export function rgbToImageData(image: RGBImage): ImageData {
  const data = new Uint8ClampedArray(image.width * image.height * 4);
  for (let i = 0, j = 0; i < image.data.length; i += 3, j += 4) {
    data[j] = image.data[i] as number;
    data[j + 1] = image.data[i + 1] as number;
    data[j + 2] = image.data[i + 2] as number;
    data[j + 3] = 255;
  }
  return new ImageData(data, image.width, image.height);
}
