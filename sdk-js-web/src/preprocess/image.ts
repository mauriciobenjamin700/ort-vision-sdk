/**
 * Composable image preprocessing primitives (resize, normalize, letterbox, layout).
 *
 * Mirrors the Python `preprocess.image` module. Operates on the canonical
 * {@link RGBImage} (HWC RGB uint8) and produces uint8 or float32 buffers
 * depending on the operation.
 */

import * as ortRuntime from "onnxruntime-web";
import type * as ort from "onnxruntime-web";

import {
  createCanvas,
  get2DContext,
  imageDataToRGB,
  rgbToImageData,
} from "../core/canvas.js";
import { RGBImage } from "../types.js";

/** Resize an image to `(targetWidth, targetHeight)` using high-quality canvas resampling. */
export function resize(
  image: RGBImage,
  targetWidth: number,
  targetHeight: number,
): RGBImage {
  if (targetWidth <= 0 || targetHeight <= 0) {
    throw new Error(`Invalid resize target ${targetWidth}x${targetHeight}.`);
  }
  if (targetWidth === image.width && targetHeight === image.height) {
    return image;
  }

  const srcCanvas = createCanvas(image.width, image.height);
  const srcCtx = get2DContext(srcCanvas);
  srcCtx.putImageData(rgbToImageData(image), 0, 0);

  const dstCanvas = createCanvas(targetWidth, targetHeight);
  const dstCtx = get2DContext(dstCanvas);
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.imageSmoothingQuality = "high";
  dstCtx.drawImage(srcCanvas as CanvasImageSource, 0, 0, targetWidth, targetHeight);

  const imageData = dstCtx.getImageData(0, 0, targetWidth, targetHeight);
  return imageDataToRGB(imageData);
}

/**
 * Convert a uint8 image to a normalized float32 array (HWC layout preserved).
 *
 * Applies `(pixel * scale - mean) / std` channel-wise.
 */
export function normalize(
  image: RGBImage,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
  scale: number = 1 / 255,
): Float32Array {
  const out = new Float32Array(image.data.length);
  const data = image.data;
  const m0 = mean[0];
  const m1 = mean[1];
  const m2 = mean[2];
  const s0 = std[0];
  const s1 = std[1];
  const s2 = std[2];
  for (let i = 0; i < data.length; i += 3) {
    out[i] = ((data[i] as number) * scale - m0) / s0;
    out[i + 1] = ((data[i + 1] as number) * scale - m1) / s1;
    out[i + 2] = ((data[i + 2] as number) * scale - m2) / s2;
  }
  return out;
}

/** Convert a uint8 image to a `Float32Array` in `[0, 1]` (HWC layout preserved). */
export function toFloat32(image: RGBImage, scale: number = 1 / 255): Float32Array {
  const out = new Float32Array(image.data.length);
  const data = image.data;
  for (let i = 0; i < data.length; i++) {
    out[i] = (data[i] as number) * scale;
  }
  return out;
}

/**
 * Transpose interleaved HWC data to planar CHW layout.
 *
 * @param hwc Source array of length `width * height * channels`.
 */
export function toCHW(
  hwc: Float32Array,
  width: number,
  height: number,
  channels: number = 3,
): Float32Array {
  const expected = width * height * channels;
  if (hwc.length !== expected) {
    throw new Error(
      `toCHW: expected length ${expected} for ${width}x${height}x${channels}, got ${hwc.length}.`,
    );
  }
  const chw = new Float32Array(expected);
  const plane = width * height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const hwcBase = (y * width + x) * channels;
      const planeIdx = y * width + x;
      for (let c = 0; c < channels; c++) {
        chw[c * plane + planeIdx] = hwc[hwcBase + c] as number;
      }
    }
  }
  return chw;
}

/** Wrap a Float32 buffer into an `ort.Tensor`. */
export function toFloat32Tensor(
  data: Float32Array,
  dims: readonly number[],
): ort.Tensor {
  return new ortRuntime.Tensor("float32", data, dims as number[]);
}

/**
 * Convert an HWC uint8 image to a CHW `Float32Array` scaled to `[0, 1]`.
 *
 * Mirrors `torchvision.transforms.ToTensor()` semantics: HWC → CHW,
 * `uint8 → float32 / 255`. Useful as input to YOLO-style detectors that
 * don't require ImageNet normalization.
 *
 * @returns CHW `Float32Array` of length `width * height * 3`.
 */
export function toTensor(image: RGBImage): Float32Array {
  const f32 = toFloat32(image);
  return toCHW(f32, image.width, image.height, 3);
}

/**
 * Convert an HWC BGR uint8 buffer (OpenCV layout) to the SDK's HWC RGB.
 *
 * Use when you receive image bytes from `cv2.imencode` over the wire and
 * want to feed them to the SDK without going through a canvas decode.
 *
 * @param bgr Flat BGR Uint8Array of length `width * height * 3`.
 */
export function fromCv2(bgr: Uint8Array, width: number, height: number): RGBImage {
  if (bgr.length !== width * height * 3) {
    throw new Error(
      `fromCv2: data length ${bgr.length} does not match width * height * 3 = ${width * height * 3}.`,
    );
  }
  const rgb = new Uint8Array(bgr.length);
  for (let i = 0; i < bgr.length; i += 3) {
    rgb[i] = bgr[i + 2] as number;
    rgb[i + 1] = bgr[i + 1] as number;
    rgb[i + 2] = bgr[i] as number;
  }
  return new RGBImage(rgb, width, height);
}

/**
 * Convert the SDK's HWC RGB image to an HWC BGR `Uint8Array` (OpenCV layout).
 *
 * Useful for round-tripping data to a Python OpenCV consumer.
 */
export function toCv2(image: RGBImage): Uint8Array {
  const rgb = image.data;
  const bgr = new Uint8Array(rgb.length);
  for (let i = 0; i < rgb.length; i += 3) {
    bgr[i] = rgb[i + 2] as number;
    bgr[i + 1] = rgb[i + 1] as number;
    bgr[i + 2] = rgb[i] as number;
  }
  return bgr;
}

export interface LetterboxResult {
  /** The padded image at the target size. */
  readonly image: RGBImage;
  /** The factor applied to the original image (`< 1` if downscaled). */
  readonly scale: number;
  /** Horizontal padding in pixels (left side; right side has the same or +1). */
  readonly padLeft: number;
  /** Vertical padding in pixels (top side). */
  readonly padTop: number;
}

/**
 * Resize preserving aspect ratio, padding to `(targetWidth, targetHeight)`
 * with a constant fill color.
 *
 * Standard YOLO preprocessing — returning `scale` and `padLeft`/`padTop`
 * lets callers map detections back to the original image coordinates.
 */
export function letterbox(
  image: RGBImage,
  targetWidth: number,
  targetHeight: number,
  fill: readonly [number, number, number] = [114, 114, 114],
): LetterboxResult {
  const scale = Math.min(targetWidth / image.width, targetHeight / image.height);
  const newW = Math.round(image.width * scale);
  const newH = Math.round(image.height * scale);
  const resized = resize(image, newW, newH);

  const out = new Uint8Array(targetWidth * targetHeight * 3);
  const f0 = fill[0];
  const f1 = fill[1];
  const f2 = fill[2];
  for (let i = 0; i < out.length; i += 3) {
    out[i] = f0;
    out[i + 1] = f1;
    out[i + 2] = f2;
  }

  const padLeft = Math.floor((targetWidth - newW) / 2);
  const padTop = Math.floor((targetHeight - newH) / 2);
  const rowBytes = newW * 3;
  for (let y = 0; y < newH; y++) {
    const srcOffset = y * rowBytes;
    const dstOffset = ((padTop + y) * targetWidth + padLeft) * 3;
    out.set(resized.data.subarray(srcOffset, srcOffset + rowBytes), dstOffset);
  }

  return {
    image: new RGBImage(out, targetWidth, targetHeight),
    scale,
    padLeft,
    padTop,
  };
}
