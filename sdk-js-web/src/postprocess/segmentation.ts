/**
 * Segmentation head postprocessing: YOLO instance-segmentation decoding.
 *
 * Compatible with YOLOv8-seg / YOLOv11-seg exports (and any later seg head
 * sharing the layout). The model produces two output tensors:
 *
 * - `output0` of shape `(1, 4 + numClasses + numMaskCoefs, numAnchors)` — the
 *   same per-anchor predictions as plain YOLO detection plus an extra
 *   `numMaskCoefs` (typically 32) channels of mask coefficients.
 * - `output1` of shape `(1, numMaskCoefs, maskH, maskW)` — a set of "prototype"
 *   masks shared across all anchors.
 *
 * The per-anchor decode (xywh→xyxy, undo letterbox, per-class NMS, sort & cap)
 * is delegated to {@link decodeYoloAnchors}; this module only handles the
 * mask-specific work: matmul against prototypes, sigmoid, bilinear resize and
 * thresholding.
 */

import { decodeYoloAnchors } from "./detection.js";
import { BoundingBox, Mask } from "../types.js";

export interface DecodeYoloSegOptions {
  readonly numClasses: number;
  /** Model input `[width, height]` (post-letterbox). */
  readonly inputWidth: number;
  readonly inputHeight: number;
  /** Original image `[width, height]`. */
  readonly originalWidth: number;
  readonly originalHeight: number;
  /** Letterbox horizontal padding in input-tensor pixels. */
  readonly padLeft: number;
  /** Letterbox vertical padding in input-tensor pixels. */
  readonly padTop: number;
  /** Letterbox scale factor. */
  readonly scale: number;
  readonly confThreshold: number;
  readonly iouThreshold: number;
  readonly maxDetections: number;
  /** Probability cutoff applied to the soft mask. Defaults to `0.5`. */
  readonly maskThreshold?: number;
}

export interface DecodedSegmentation {
  readonly bbox: BoundingBox;
  readonly classId: number;
  readonly confidence: number;
  /** Binary mask cropped to `bbox`. Width/height match `bbox.asIntXyxy()` extents. */
  readonly mask: Mask;
}

/**
 * Decode YOLO segmentation raw outputs into a list of segmented instances.
 *
 * Compatible with YOLOv8-seg / YOLOv11-seg.
 *
 * The prototypes' batch axis is stripped here; the per-anchor batch axis is
 * validated by {@link decodeYoloAnchors}. Each instance's box is mapped back
 * into input-tensor coordinates and from there into the low-resolution
 * prototype grid, and only the prototype pixels under it are assembled.
 *
 * @param perAnchorData Flat `output0`, length `(4 + numClasses + numMaskCoefs) * numAnchors`.
 * @param perAnchorDims Dims as reported by ORT, e.g. `[1, 116, 8400]`.
 * @param prototypeData Flat `output1`, length `numMaskCoefs * maskH * maskW`.
 * @param prototypeDims Dims as reported by ORT, e.g. `[1, 32, 160, 160]`.
 */
export function decodeYoloSeg(
  perAnchorData: Float32Array,
  perAnchorDims: readonly number[],
  prototypeData: Float32Array,
  prototypeDims: readonly number[],
  options: DecodeYoloSegOptions,
): DecodedSegmentation[] {
  let pDims = prototypeDims;
  if (pDims.length === 4) {
    if (pDims[0] !== 1) {
      throw new Error(`decodeYoloSeg: expected batch size 1 in prototypes, got ${pDims[0]}.`);
    }
    pDims = [pDims[1] as number, pDims[2] as number, pDims[3] as number];
  }
  if (pDims.length !== 3) {
    throw new Error(
      `decodeYoloSeg: expected 3-D prototypes after batch removal, got dims=${JSON.stringify(prototypeDims)}.`,
    );
  }
  const numMaskCoefs = pDims[0] as number;
  const maskH = pDims[1] as number;
  const maskW = pDims[2] as number;

  const channels = perAnchorDims.length === 3 ? perAnchorDims[1] : perAnchorDims[0];
  const numAnchors = perAnchorDims.length === 3 ? perAnchorDims[2] : perAnchorDims[1];
  if (channels === undefined || numAnchors === undefined) {
    throw new Error(
      `decodeYoloSeg: cannot read channels/numAnchors from dims=${JSON.stringify(perAnchorDims)}.`,
    );
  }

  const expectedChannels = 4 + options.numClasses + numMaskCoefs;
  if (channels !== expectedChannels) {
    throw new Error(
      `decodeYoloSeg: channels=${channels} does not match 4 + numClasses(${options.numClasses}) + numMaskCoefs(${numMaskCoefs}) = ${expectedChannels}.`,
    );
  }
  if (prototypeData.length !== numMaskCoefs * maskH * maskW) {
    throw new Error(
      `decodeYoloSeg: prototype length ${prototypeData.length} does not match dims=${JSON.stringify(prototypeDims)}.`,
    );
  }

  const decoded = decodeYoloAnchors(perAnchorData, perAnchorDims, {
    numClasses: options.numClasses,
    originalWidth: options.originalWidth,
    originalHeight: options.originalHeight,
    padLeft: options.padLeft,
    padTop: options.padTop,
    scale: options.scale,
    confThreshold: options.confThreshold,
    iouThreshold: options.iouThreshold,
    maxDetections: options.maxDetections,
  });

  if (decoded.anchorIndices.length === 0) return [];

  const maskThreshold = options.maskThreshold ?? 0.5;
  const scaleX = maskW / options.inputWidth;
  const scaleY = maskH / options.inputHeight;
  const protoPlane = maskH * maskW;
  const coefBase = (4 + options.numClasses) * numAnchors;

  const results: DecodedSegmentation[] = [];

  for (let i = 0; i < decoded.anchorIndices.length; i++) {
    const a = decoded.anchorIndices[i] as number;
    const x1 = decoded.boxesXyxy[i * 4] as number;
    const y1 = decoded.boxesXyxy[i * 4 + 1] as number;
    const x2 = decoded.boxesXyxy[i * 4 + 2] as number;
    const y2 = decoded.boxesXyxy[i * 4 + 3] as number;
    const bbox = new BoundingBox(x1, y1, x2, y2);
    const classId = decoded.classIds[i] as number;
    const confidence = decoded.confidences[i] as number;

    const bboxW = Math.max(0, Math.trunc(x2) - Math.trunc(x1));
    const bboxH = Math.max(0, Math.trunc(y2) - Math.trunc(y1));

    if (bboxW === 0 || bboxH === 0) {
      results.push({ bbox, classId, confidence, mask: new Mask(new Uint8Array(0), 0, 0) });
      continue;
    }

    const ibx1 = x1 * options.scale + options.padLeft;
    const iby1 = y1 * options.scale + options.padTop;
    const ibx2 = x2 * options.scale + options.padLeft;
    const iby2 = y2 * options.scale + options.padTop;

    const mbx1 = Math.max(0, Math.floor(ibx1 * scaleX));
    const mby1 = Math.max(0, Math.floor(iby1 * scaleY));
    const mbx2 = Math.min(maskW, Math.ceil(ibx2 * scaleX));
    const mby2 = Math.min(maskH, Math.ceil(iby2 * scaleY));

    if (mbx2 <= mbx1 || mby2 <= mby1) {
      results.push({
        bbox,
        classId,
        confidence,
        mask: new Mask(new Uint8Array(bboxW * bboxH), bboxW, bboxH),
      });
      continue;
    }

    const cropW = mbx2 - mbx1;
    const cropH = mby2 - mby1;
    const softCrop = softMaskCrop(
      perAnchorData,
      prototypeData,
      coefBase + a,
      numAnchors,
      numMaskCoefs,
      maskW,
      protoPlane,
      mbx1,
      mby1,
      cropW,
      cropH,
    );
    const binary = resizeBilinearThreshold(
      softCrop,
      cropW,
      cropH,
      bboxW,
      bboxH,
      maskThreshold,
    );

    results.push({ bbox, classId, confidence, mask: new Mask(binary, bboxW, bboxH) });
  }

  return results;
}

function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

/**
 * Assemble one instance's soft mask over the prototype region under its box.
 *
 * The loop runs coefficient-major: for each of the `numMaskCoefs` prototypes it
 * walks the crop row by row, so every read of `prototypeData` is contiguous.
 * The pixel-major order it replaces jumped a whole prototype plane (25 600
 * floats for a 160×160 head) between consecutive reads, touching a fresh cache
 * line on every multiply. The accumulator is a `Float64Array` so each pixel's
 * sum is still built in double precision and in the same coefficient order as a
 * scalar `let sum = 0` — the result is bit-identical, only the memory walk
 * changed.
 *
 * @param perAnchorData Flat `output0`.
 * @param prototypeData Flat `output1`.
 * @param coefOffset Index of this anchor's first mask coefficient.
 * @param numAnchors Stride between consecutive coefficients of one anchor.
 * @param numMaskCoefs Number of prototypes.
 * @param maskW Prototype width.
 * @param protoPlane Prototype plane size, `maskW * maskH`.
 * @param left First prototype column of the crop.
 * @param top First prototype row of the crop.
 * @param cropW Crop width in prototype pixels.
 * @param cropH Crop height in prototype pixels.
 * @returns The sigmoid of the mask logits, `cropW * cropH`, row-major.
 */
function softMaskCrop(
  perAnchorData: Float32Array,
  prototypeData: Float32Array,
  coefOffset: number,
  numAnchors: number,
  numMaskCoefs: number,
  maskW: number,
  protoPlane: number,
  left: number,
  top: number,
  cropW: number,
  cropH: number,
): Float32Array {
  const acc = new Float64Array(cropW * cropH);
  for (let kk = 0; kk < numMaskCoefs; kk++) {
    const coef = perAnchorData[coefOffset + kk * numAnchors] as number;
    const planeBase = kk * protoPlane + left;
    for (let y = 0; y < cropH; y++) {
      const src = planeBase + (top + y) * maskW;
      const dst = y * cropW;
      for (let x = 0; x < cropW; x++) {
        acc[dst + x] = (acc[dst + x] as number) + coef * (prototypeData[src + x] as number);
      }
    }
  }
  const soft = new Float32Array(acc.length);
  for (let j = 0; j < acc.length; j++) soft[j] = sigmoid(acc[j] as number);
  return soft;
}

/**
 * Bilinearly resize a soft mask and threshold it in the same pass.
 *
 * Half-pixel-centre sampling, matching `cv2.resize(..., INTER_LINEAR)` and the
 * Python SDK. The column taps and weights depend only on `x`, so they are
 * computed once instead of once per row, and the thresholded byte is written
 * directly instead of materialising a `Float32Array` of the resized mask first.
 * Each interpolated value is rounded to float32 with `Math.fround` before the
 * comparison, because that is the value the intermediate array used to hold —
 * comparing the unrounded double would flip pixels sitting exactly on the
 * threshold.
 *
 * @param src Soft mask, `srcWidth * srcHeight`, row-major.
 * @param srcWidth Source width.
 * @param srcHeight Source height.
 * @param targetWidth Output width, at least 1.
 * @param targetHeight Output height, at least 1.
 * @param threshold Probability at or above which a pixel is foreground.
 * @returns `255` for foreground and `0` for background, `targetWidth * targetHeight`.
 */
function resizeBilinearThreshold(
  src: Float32Array,
  srcWidth: number,
  srcHeight: number,
  targetWidth: number,
  targetHeight: number,
  threshold: number,
): Uint8Array {
  const out = new Uint8Array(targetWidth * targetHeight);
  if (targetWidth === srcWidth && targetHeight === srcHeight) {
    for (let j = 0; j < out.length; j++) out[j] = (src[j] as number) >= threshold ? 255 : 0;
    return out;
  }
  const sx = srcWidth / targetWidth;
  const sy = srcHeight / targetHeight;
  const x0s = new Int32Array(targetWidth);
  const x1s = new Int32Array(targetWidth);
  const wxs = new Float64Array(targetWidth);
  for (let x = 0; x < targetWidth; x++) {
    const xx = (x + 0.5) * sx - 0.5;
    const x0 = Math.max(0, Math.floor(xx));
    x0s[x] = x0;
    x1s[x] = Math.min(srcWidth - 1, x0 + 1);
    wxs[x] = Math.max(0, Math.min(1, xx - x0));
  }
  for (let y = 0; y < targetHeight; y++) {
    const yy = (y + 0.5) * sy - 0.5;
    const y0 = Math.max(0, Math.floor(yy));
    const y1 = Math.min(srcHeight - 1, y0 + 1);
    const wy = Math.max(0, Math.min(1, yy - y0));
    const row0 = y0 * srcWidth;
    const row1 = y1 * srcWidth;
    const dst = y * targetWidth;
    for (let x = 0; x < targetWidth; x++) {
      const x0 = x0s[x] as number;
      const x1 = x1s[x] as number;
      const wx = wxs[x] as number;
      const top = (src[row0 + x0] as number) * (1 - wx) + (src[row0 + x1] as number) * wx;
      const bot = (src[row1 + x0] as number) * (1 - wx) + (src[row1 + x1] as number) * wx;
      out[dst + x] = Math.fround(top * (1 - wy) + bot * wy) >= threshold ? 255 : 0;
    }
  }
  return out;
}
