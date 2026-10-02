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
  imageDataToRGBChecked,
} from "../core/canvas.js";
import { ImageLoadError } from "../core/exceptions.js";
import { RGBImage } from "../types.js";

/** Anything {@link loadImage} accepts as an image input. */
export type ImageInput =
  | string
  | Blob
  | HTMLImageElement
  | HTMLVideoElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageBitmap
  | VideoFrame
  | ImageData
  | RGBImage;

/**
 * A decoded input, plus what the tasks can reuse from decoding it.
 *
 * Internal to the SDK: the tasks need more than {@link loadImage} returns, and
 * the extra fields are about avoiding work, not about the image itself.
 */
export interface LoadedImage {
  /** The input as the canonical HWC RGB image. */
  readonly image: RGBImage;
  /**
   * A full-resolution canvas holding exactly {@link image}'s pixels, or `null`.
   *
   * Decoding any drawable already paints it onto a canvas to read it back.
   * Handing that canvas to a preprocessing pipeline lets it `drawImage` the
   * frame straight into the model input, instead of rebuilding RGBA from
   * {@link image} and `putImageData`-ing it onto a canvas of its own — two
   * full-resolution passes and an allocation the size of the frame. Only set
   * when every pixel was opaque: a translucent pixel composites differently
   * from the RGB that dropping its alpha kept.
   */
  readonly canvas: CanvasImageSource | null;
  /**
   * Whether {@link image}'s buffer was allocated here rather than passed in.
   *
   * An `RGBImage` the caller handed over is theirs, and a video loop may well
   * refill the same buffer every frame. Anything derived from it lazily would
   * then read a later frame, so lazy crops are only taken on owned buffers.
   */
  readonly owned: boolean;
}

/**
 * Load an image from any supported source into a HWC uint8 RGB array.
 *
 * A video element must have its current frame available (`readyState >= 2`);
 * the frame showing at the time of the call is the one decoded. A `VideoFrame`
 * is read but not closed — it belongs to the caller.
 *
 * @throws {@link ImageLoadError} if the source cannot be decoded or has an unsupported shape.
 */
export async function loadImage(source: ImageInput): Promise<RGBImage> {
  return (await loadImageSource(source)).image;
}

/**
 * Decode an input like {@link loadImage}, keeping what preprocessing can reuse.
 *
 * @param source Any {@link ImageInput}.
 * @returns The image, the canvas it was painted on (when reusable), and
 *   whether the SDK owns the pixel buffer.
 * @throws {@link ImageLoadError} if the source cannot be decoded.
 */
export async function loadImageSource(source: ImageInput): Promise<LoadedImage> {
  if (source instanceof RGBImage) {
    return { image: source, canvas: null, owned: false };
  }

  if (typeof ImageData !== "undefined" && source instanceof ImageData) {
    return { image: imageDataToRGBChecked(source).image, canvas: null, owned: true };
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

  if (typeof HTMLVideoElement !== "undefined" && source instanceof HTMLVideoElement) {
    if (source.readyState < 2) {
      throw new ImageLoadError(
        "HTMLVideoElement has no current frame yet (readyState < HAVE_CURRENT_DATA). " +
          "Wait for its 'loadeddata' event before predicting on it.",
      );
    }
    return drawableToRGB(source, source.videoWidth, source.videoHeight, isLiveStream(source));
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

  if (typeof VideoFrame !== "undefined" && source instanceof VideoFrame) {
    return drawableToRGB(
      source,
      source.displayWidth,
      source.displayHeight,
      source.format !== null && OPAQUE_FRAME_FORMATS.has(source.format),
    );
  }

  throw new ImageLoadError(
    `Unsupported image source type: ${Object.prototype.toString.call(source)}`,
  );
}

async function loadFromUrl(url: string): Promise<LoadedImage> {
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

async function loadFromBlob(blob: Blob): Promise<LoadedImage> {
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
    return drawableToRGB(bitmap, bitmap.width, bitmap.height, blob.type === "image/jpeg");
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

/**
 * `VideoFrame` pixel formats that carry no alpha channel.
 *
 * The `A`-suffixed planar formats and `RGBA`/`BGRA` do; a frame in one of
 * these never has a translucent pixel.
 */
const OPAQUE_FRAME_FORMATS: ReadonlySet<string> = new Set([
  "I420",
  "I422",
  "I444",
  "NV12",
  "RGBX",
  "BGRX",
]);

/**
 * Whether a video element is playing a live capture (camera, screen).
 *
 * `MediaStream` video tracks carry no alpha, so every frame is opaque. A video
 * playing a file may not be — VP9 with alpha exists — so it gets no such
 * promise.
 *
 * @param video The element.
 */
function isLiveStream(video: HTMLVideoElement): boolean {
  return typeof MediaStream !== "undefined" && video.srcObject instanceof MediaStream;
}

/**
 * Paint a drawable onto a fresh canvas and read it back as RGB.
 *
 * When the source is known to be opaque without looking — a JPEG, a camera
 * frame, an alpha-less `VideoFrame` — the read-back is deferred: the image's
 * pixels come out of the canvas the first time something reads `data`. The
 * canvas is this call's own, so it still holds the frame of this moment
 * however late that read happens. Preprocessing draws from the canvas and
 * never reads `data`, so a loop that only consumes boxes skips the
 * full-resolution `getImageData` altogether.
 *
 * Any other source is read back now, because whether its canvas may stand in
 * for the RGB image depends on every pixel being opaque, and only the read
 * tells.
 *
 * @param drawable The source to decode.
 * @param width Its intrinsic width.
 * @param height Its intrinsic height.
 * @param knownOpaque Whether the source cannot contain a translucent pixel.
 * @returns The decoded image, with the canvas attached when it is opaque.
 * @throws {@link ImageLoadError} for a zero-sized source.
 */
function drawableToRGB(
  drawable: CanvasImageSource,
  width: number,
  height: number,
  knownOpaque: boolean = false,
): LoadedImage {
  if (width === 0 || height === 0) {
    throw new ImageLoadError(`Cannot load image with zero dimension (${width}x${height}).`);
  }
  const canvas = createCanvas(width, height);
  const ctx = get2DContext(canvas);
  ctx.drawImage(drawable, 0, 0, width, height);
  if (knownOpaque) {
    const image = RGBImage.deferred(
      width,
      height,
      () => imageDataToRGBChecked(ctx.getImageData(0, 0, width, height)).image.data,
    );
    return { image, canvas, owned: true };
  }
  const { image, opaque } = imageDataToRGBChecked(ctx.getImageData(0, 0, width, height));
  return { image, canvas: opaque ? canvas : null, owned: true };
}
