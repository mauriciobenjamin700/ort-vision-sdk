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

/**
 * Whether typed-array words are stored least-significant byte first.
 *
 * The pixel conversions below move one RGBA pixel as one `Uint32` — about a
 * quarter fewer milliseconds per 1080p frame than four byte accesses — which
 * only maps `R` to the low byte on a little-endian platform. Every browser
 * engine in use is little-endian; the byte-wise loops stay as the fallback so
 * correctness never rests on that.
 */
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
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

/**
 * Get a 2D context from a canvas, throwing a typed error if the browser refuses.
 *
 * @param canvas The canvas to obtain a context for.
 * @param options Context attributes, forwarded verbatim. `willReadFrequently`
 *   tells the browser the canvas is read back with `getImageData` repeatedly,
 *   which moves the backing store to software: faster readback, slower
 *   compositing. Whether it pays depends on the surface, so it is the caller's
 *   call rather than a default here.
 */
export function get2DContext(
  canvas: Canvas2D,
  options: CanvasRenderingContext2DSettings = {},
): Context2D {
  const ctx = canvas.getContext("2d", options) as Context2D | null;
  if (ctx === null) {
    throw new ImageLoadError("Failed to obtain 2D rendering context.");
  }
  return ctx;
}

/** Convert RGBA `ImageData` into the canonical HWC RGB `RGBImage`. */
export function imageDataToRGB(imageData: ImageData): RGBImage {
  return imageDataToRGBChecked(imageData).image;
}

/** An {@link RGBImage} plus whether the RGBA it came from was fully opaque. */
export interface CheckedRGB {
  readonly image: RGBImage;
  /** `true` when every source pixel had alpha 255. */
  readonly opaque: boolean;
}

/**
 * Convert RGBA `ImageData` into an `RGBImage`, noting whether alpha was all 255.
 *
 * Dropping alpha is lossy only when some pixel was translucent: the RGB kept is
 * the un-premultiplied colour, which {@link rgbToImageData} later paints back as
 * fully opaque. A canvas still holding the original RGBA composites that pixel
 * differently, so it can stand in for the `RGBImage` only when nothing was
 * translucent — which is what `opaque` reports. The check rides the copy loop
 * that already reads every alpha byte, so it costs one AND per pixel. The loop
 * reads a whole pixel as one `Uint32` where {@link LITTLE_ENDIAN} allows.
 *
 * @param imageData Source pixels.
 * @returns The packed RGB image and the opacity verdict.
 * @throws {@link ImageLoadError} if the buffer length does not match the size.
 */
export function imageDataToRGBChecked(imageData: ImageData): CheckedRGB {
  const { data, width, height } = imageData;
  if (data.length !== width * height * 4) {
    throw new ImageLoadError(
      `Unexpected ImageData length ${data.length} for ${width}x${height} (expected ${
        width * height * 4
      }).`,
    );
  }
  const pixels = width * height;
  const rgb = new Uint8Array(pixels * 3);
  if (LITTLE_ENDIAN && data.byteOffset % 4 === 0) {
    const words = new Uint32Array(data.buffer, data.byteOffset, pixels);
    let alpha = -1;
    for (let p = 0, j = 0; p < pixels; p++, j += 3) {
      const word = words[p] as number;
      rgb[j] = word;
      rgb[j + 1] = word >>> 8;
      rgb[j + 2] = word >>> 16;
      alpha &= word;
    }
    return { image: new RGBImage(rgb, width, height), opaque: alpha >>> 24 === 255 };
  }
  let alpha = 255;
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i] as number;
    rgb[j + 1] = data[i + 1] as number;
    rgb[j + 2] = data[i + 2] as number;
    alpha &= data[i + 3] as number;
  }
  return { image: new RGBImage(rgb, width, height), opaque: alpha === 255 };
}

/**
 * Convert an `RGBImage` into RGBA `ImageData` (alpha forced to 255).
 *
 * Writes one 32-bit word per pixel where the platform allows — see
 * {@link LITTLE_ENDIAN} — instead of four separate byte stores.
 */
export function rgbToImageData(image: RGBImage): ImageData {
  const pixels = image.width * image.height;
  const data = new Uint8ClampedArray(pixels * 4);
  const src = image.data;
  if (LITTLE_ENDIAN) {
    const words = new Uint32Array(data.buffer);
    for (let p = 0, i = 0; p < pixels; p++, i += 3) {
      words[p] =
        ((src[i] as number) |
          ((src[i + 1] as number) << 8) |
          ((src[i + 2] as number) << 16) |
          0xff000000) >>>
        0;
    }
    return new ImageData(data, image.width, image.height);
  }
  for (let i = 0, j = 0; i < image.data.length; i += 3, j += 4) {
    data[j] = image.data[i] as number;
    data[j + 1] = image.data[i + 1] as number;
    data[j + 2] = image.data[i + 2] as number;
    data[j + 3] = 255;
  }
  return new ImageData(data, image.width, image.height);
}
