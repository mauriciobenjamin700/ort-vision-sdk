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
  // Strip batch dim from prototypes; perAnchor batch is validated by the helper.
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

    // Bbox in input-tensor coords, then in low-res mask coords.
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

    // Compute soft mask only within the prototype region under this bbox.
    const cropW = mbx2 - mbx1;
    const cropH = mby2 - mby1;
    const softCrop = new Float32Array(cropW * cropH);
    for (let y = 0; y < cropH; y++) {
      const py = mby1 + y;
      for (let x = 0; x < cropW; x++) {
        const px = mbx1 + x;
        let sum = 0;
        for (let kk = 0; kk < numMaskCoefs; kk++) {
          const coef = perAnchorData[coefBase + kk * numAnchors + a];
          const proto = prototypeData[kk * protoPlane + py * maskW + px];
          sum += coef * proto;
        }
        softCrop[y * cropW + x] = sigmoid(sum);
      }
    }

    const resized = resizeBilinear(softCrop, cropW, cropH, bboxW, bboxH);
    const binary = new Uint8Array(bboxW * bboxH);
    for (let j = 0; j < binary.length; j++) {
      binary[j] = resized[j] >= maskThreshold ? 255 : 0;
    }

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

let _warnedDecodeYoloV8Seg = false;

/** @deprecated since 0.2.0 — use {@link decodeYoloSeg}. Will be removed in 0.4.0. */
export function decodeYoloV8Seg(
  perAnchorData: Float32Array,
  perAnchorDims: readonly number[],
  prototypeData: Float32Array,
  prototypeDims: readonly number[],
  options: DecodeYoloSegOptions,
): DecodedSegmentation[] {
  if (!_warnedDecodeYoloV8Seg) {
    _warnedDecodeYoloV8Seg = true;
    console.warn(
      "[@ort-vision-sdk/web] decodeYoloV8Seg is deprecated since 0.2.0; use decodeYoloSeg. " +
        "The alias will be removed in 0.4.0.",
    );
  }
  return decodeYoloSeg(
    perAnchorData,
    perAnchorDims,
    prototypeData,
    prototypeDims,
    options,
  );
}

/** @deprecated since 0.2.0 — use {@link DecodeYoloSegOptions}. */
export type DecodeYoloV8SegOptions = DecodeYoloSegOptions;

function resizeBilinear(
  src: Float32Array,
  srcWidth: number,
  srcHeight: number,
  targetWidth: number,
  targetHeight: number,
): Float32Array {
  const out = new Float32Array(targetWidth * targetHeight);
  if (targetWidth === 0 || targetHeight === 0 || srcWidth === 0 || srcHeight === 0) {
    return out;
  }
  if (targetWidth === srcWidth && targetHeight === srcHeight) {
    out.set(src);
    return out;
  }
  const sx = srcWidth / targetWidth;
  const sy = srcHeight / targetHeight;
  for (let y = 0; y < targetHeight; y++) {
    const yy = (y + 0.5) * sy - 0.5;
    const y0 = Math.max(0, Math.floor(yy));
    const y1 = Math.min(srcHeight - 1, y0 + 1);
    const wy = Math.max(0, Math.min(1, yy - y0));
    for (let x = 0; x < targetWidth; x++) {
      const xx = (x + 0.5) * sx - 0.5;
      const x0 = Math.max(0, Math.floor(xx));
      const x1 = Math.min(srcWidth - 1, x0 + 1);
      const wx = Math.max(0, Math.min(1, xx - x0));
      const v00 = src[y0 * srcWidth + x0];
      const v01 = src[y0 * srcWidth + x1];
      const v10 = src[y1 * srcWidth + x0];
      const v11 = src[y1 * srcWidth + x1];
      const top = v00 * (1 - wx) + v01 * wx;
      const bot = v10 * (1 - wx) + v11 * wx;
      out[y * targetWidth + x] = top * (1 - wy) + bot * wy;
    }
  }
  return out;
}
