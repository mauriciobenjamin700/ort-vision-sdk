/**
 * Image loading utilities — convert any supported source into the canonical
 * {@link RGBImage} format.
 *
 * Browser-only: uses `fetch`, `createImageBitmap`, and a Canvas 2D context to
 * decode arbitrary inputs into a tightly-packed `Uint8Array` of HWC RGB pixels.
 */

import {
  createCanvas,
  get2DContext,
  imageDataToRGB,
} from "../core/canvas.js";
import { ImageLoadError } from "../core/exceptions.js";
import { RGBImage } from "../types.js";

/** Anything {@link loadImage} accepts as an image input. */
export type ImageInput =
  | string
  | Blob
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageBitmap
  | ImageData
  | RGBImage;

/**
 * Load an image from any supported source into a HWC uint8 RGB array.
 *
 * @throws {@link ImageLoadError} if the source cannot be decoded or has an unsupported shape.
 */
export async function loadImage(source: ImageInput): Promise<RGBImage> {
  if (source instanceof RGBImage) {
    return source;
  }

  if (typeof ImageData !== "undefined" && source instanceof ImageData) {
    return imageDataToRGB(source);
  }

  if (typeof source === "string") {
    return loadFromUrl(source);
  }

  if (typeof Blob !== "undefined" && source instanceof Blob) {
    return loadFromBlob(source);
  }

  if (typeof HTMLImageElement !== "undefined" && source instanceof HTMLImageElement) {
    await waitForImageElement(source);
    return drawableToRGB(source, source.naturalWidth, source.naturalHeight);
  }

  if (typeof HTMLCanvasElement !== "undefined" && source instanceof HTMLCanvasElement) {
    return drawableToRGB(source, source.width, source.height);
  }

  if (typeof OffscreenCanvas !== "undefined" && source instanceof OffscreenCanvas) {
    return drawableToRGB(source, source.width, source.height);
  }

  if (typeof ImageBitmap !== "undefined" && source instanceof ImageBitmap) {
    return drawableToRGB(source, source.width, source.height);
  }

  throw new ImageLoadError(
    `Unsupported image source type: ${Object.prototype.toString.call(source)}`,
  );
}

async function loadFromUrl(url: string): Promise<RGBImage> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (err) {
    throw new ImageLoadError(
      `Failed to fetch image from ${url}: ${(err as Error).message}`,
      { cause: err },
    );
  }
  if (!response.ok) {
    throw new ImageLoadError(
      `Failed to fetch image from ${url}: HTTP ${response.status} ${response.statusText}`,
    );
  }
  const blob = await response.blob();
  return loadFromBlob(blob);
}

async function loadFromBlob(blob: Blob): Promise<RGBImage> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(blob);
  } catch (err) {
    throw new ImageLoadError(
      `Failed to decode image blob: ${(err as Error).message}`,
      { cause: err },
    );
  }
  try {
    return drawableToRGB(bitmap, bitmap.width, bitmap.height);
  } finally {
    bitmap.close();
  }
}

function waitForImageElement(img: HTMLImageElement): Promise<void> {
  if (img.complete && img.naturalWidth > 0) {
    return Promise.resolve();
  }
  return new Promise<void>((resolve, reject) => {
    const onLoad = (): void => {
      cleanup();
      resolve();
    };
    const onError = (): void => {
      cleanup();
      reject(new ImageLoadError("Failed to load HTMLImageElement (load event errored)"));
    };
    const cleanup = (): void => {
      img.removeEventListener("load", onLoad);
      img.removeEventListener("error", onError);
    };
    img.addEventListener("load", onLoad, { once: true });
    img.addEventListener("error", onError, { once: true });
  });
}

function drawableToRGB(
  drawable: CanvasImageSource,
  width: number,
  height: number,
): RGBImage {
  if (width === 0 || height === 0) {
    throw new ImageLoadError(`Cannot load image with zero dimension (${width}x${height}).`);
  }
  const canvas = createCanvas(width, height);
  const ctx = get2DContext(canvas);
  ctx.drawImage(drawable, 0, 0);
  const imageData = ctx.getImageData(0, 0, width, height);
  return imageDataToRGB(imageData);
}
