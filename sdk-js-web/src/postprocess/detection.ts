/**
 * Detection head postprocessing: anchor-free YOLO decoding + non-maximum suppression.
 *
 * The shared {@link decodeYoloAnchors} helper does the per-anchor work that
 * is identical for plain detection and segmentation (transpose, xywh→xyxy,
 * letterbox unmap, per-class NMS, sort & cap). {@link decodeYolo} is a thin
 * wrapper around it; the segmentation module ({@link ./segmentation.js})
 * calls the helper directly so it can also recover the per-anchor mask
 * coefficients.
 *
 * Works for any YOLO export with the post-v8 anchor-free head:
 * **YOLOv8 / v9 / v10 / v11 / v12** detect heads, all of which share the
 * `[1, 4 + nc, N]` output layout.
 */

import { BoundingBox } from "../types.js";

/**
 * Greedy non-maximum suppression on axis-aligned bounding boxes.
 *
 * Mirrors `torchvision.ops.nms` (keeps boxes with the highest score, drops
 * any subsequent box whose IoU exceeds the threshold).
 *
 * @param boxes Flat array of length `4 * N` in xyxy order: `[x1,y1,x2,y2, ...]`.
 * @param scores Detection score per box, length `N`.
 * @param iouThreshold Boxes with IoU above this threshold relative to a kept box are suppressed.
 * @returns Indices of kept boxes, in descending score order.
 */
export function nms(
  boxes: Float32Array,
  scores: Float32Array,
  iouThreshold: number,
): Int32Array {
  const n = scores.length;
  if (n === 0) return new Int32Array(0);

  const areas = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x1 = boxes[i * 4] as number;
    const y1 = boxes[i * 4 + 1] as number;
    const x2 = boxes[i * 4 + 2] as number;
    const y2 = boxes[i * 4 + 3] as number;
    areas[i] = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  }

  const order = new Array<number>(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => (scores[b] as number) - (scores[a] as number));

  const suppressed = new Uint8Array(n);
  const keep: number[] = [];

  for (let oi = 0; oi < order.length; oi++) {
    const i = order[oi] as number;
    if (suppressed[i]) continue;
    keep.push(i);

    const ax1 = boxes[i * 4] as number;
    const ay1 = boxes[i * 4 + 1] as number;
    const ax2 = boxes[i * 4 + 2] as number;
    const ay2 = boxes[i * 4 + 3] as number;
    const ai = areas[i] as number;

    for (let oj = oi + 1; oj < order.length; oj++) {
      const j = order[oj] as number;
      if (suppressed[j]) continue;

      const bx1 = boxes[j * 4] as number;
      const by1 = boxes[j * 4 + 1] as number;
      const bx2 = boxes[j * 4 + 2] as number;
      const by2 = boxes[j * 4 + 3] as number;

      const ix1 = Math.max(ax1, bx1);
      const iy1 = Math.max(ay1, by1);
      const ix2 = Math.min(ax2, bx2);
      const iy2 = Math.min(ay2, by2);
      const iw = Math.max(0, ix2 - ix1);
      const ih = Math.max(0, iy2 - iy1);
      const inter = iw * ih;
      const union = ai + (areas[j] as number) - inter;
      const iou = union > 0 ? inter / union : 0;
      if (iou > iouThreshold) suppressed[j] = 1;
    }
  }

  return Int32Array.from(keep);
}

/**
 * Per-class NMS — boxes are suppressed only by other boxes of the same class.
 *
 * Mirrors `torchvision.ops.batched_nms`.
 *
 * @param boxes Flat array of length `4 * N` in xyxy order.
 * @param scores Detection score per box, length `N`.
 * @param idxs Class index per box, length `N`. Boxes with different `idxs`
 *   never suppress each other.
 * @param iouThreshold IoU threshold for suppression within a class.
 */
export function batchedNms(
  boxes: Float32Array,
  scores: Float32Array,
  idxs: Int32Array,
  iouThreshold: number,
): Int32Array {
  if (scores.length === 0) return new Int32Array(0);

  const byClass = new Map<number, number[]>();
  for (let i = 0; i < idxs.length; i++) {
    const c = idxs[i] as number;
    const list = byClass.get(c);
    if (list === undefined) byClass.set(c, [i]);
    else list.push(i);
  }

  const keep: number[] = [];
  for (const indices of byClass.values()) {
    const m = indices.length;
    const subBoxes = new Float32Array(m * 4);
    const subScores = new Float32Array(m);
    for (let k = 0; k < m; k++) {
      const i = indices[k] as number;
      subBoxes[k * 4] = boxes[i * 4] as number;
      subBoxes[k * 4 + 1] = boxes[i * 4 + 1] as number;
      subBoxes[k * 4 + 2] = boxes[i * 4 + 2] as number;
      subBoxes[k * 4 + 3] = boxes[i * 4 + 3] as number;
      subScores[k] = scores[i] as number;
    }
    const subKeep = nms(subBoxes, subScores, iouThreshold);
    for (let k = 0; k < subKeep.length; k++) {
      keep.push(indices[subKeep[k] as number] as number);
    }
  }

  keep.sort((a, b) => (scores[b] as number) - (scores[a] as number));
  return Int32Array.from(keep);
}

export interface DecodeYoloAnchorsOptions {
  /** Number of class-score channels following the 4 box channels. */
  readonly numClasses: number;
  readonly originalWidth: number;
  readonly originalHeight: number;
  readonly padLeft: number;
  readonly padTop: number;
  readonly scale: number;
  readonly confThreshold: number;
  readonly iouThreshold: number;
  readonly maxDetections: number;
}

export interface DecodedAnchors {
  /** Indices into the original `numAnchors` axis, in descending confidence order. */
  readonly anchorIndices: Int32Array;
  /** `[k, 4]` boxes in original-image pixel coords, flat row-major xyxy. */
  readonly boxesXyxy: Float32Array;
  /** Predicted class id per survivor. */
  readonly classIds: Int32Array;
  /** Confidence per survivor. */
  readonly confidences: Float32Array;
}

/**
 * Shared YOLO per-anchor decode used by both detection and segmentation
 * (v8 / v9 / v10 / v11 / v12).
 *
 * Only the first `4 + numClasses` channels are read; later channels (e.g.
 * mask coefficients) are ignored — callers can fetch them via the returned
 * {@link DecodedAnchors.anchorIndices}.
 *
 * @param data Flat per-anchor output, length `channels * numAnchors`.
 * @param dims Dims as reported by ORT, e.g. `[1, 84, 8400]` (det) or
 *   `[1, 116, 8400]` (seg). The leading batch dim must be 1.
 */
export function decodeYoloAnchors(
  data: Float32Array,
  dims: readonly number[],
  options: DecodeYoloAnchorsOptions,
): DecodedAnchors {
  let normalized = dims;
  if (normalized.length === 3) {
    if (normalized[0] !== 1) {
      throw new Error(`decodeYoloAnchors: expected batch size 1, got ${normalized[0]}.`);
    }
    normalized = [normalized[1] as number, normalized[2] as number];
  }
  if (normalized.length !== 2) {
    throw new Error(
      `decodeYoloAnchors: expected 2-D output after batch removal, got dims=${JSON.stringify(dims)}.`,
    );
  }
  const channels = normalized[0] as number;
  const numAnchors = normalized[1] as number;

  const {
    numClasses,
    originalWidth,
    originalHeight,
    padLeft,
    padTop,
    scale,
    confThreshold,
    iouThreshold,
    maxDetections,
  } = options;

  if (numClasses < 1 || numClasses + 4 > channels) {
    throw new Error(
      `decodeYoloAnchors: invalid numClasses=${numClasses} for channels=${channels}.`,
    );
  }
  if (data.length !== channels * numAnchors) {
    throw new Error(
      `decodeYoloAnchors: data length ${data.length} does not match channels*numAnchors=${channels * numAnchors}.`,
    );
  }

  type Candidate = {
    anchorIdx: number;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    classId: number;
    confidence: number;
  };
  const candidates: Candidate[] = [];

  for (let a = 0; a < numAnchors; a++) {
    let bestCls = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const s = data[(4 + c) * numAnchors + a];
      if (s !== undefined && s > bestScore) {
        bestScore = s;
        bestCls = c;
      }
    }
    if (bestScore < confThreshold) continue;

    const cx = data[a] as number;
    const cy = data[numAnchors + a] as number;
    const w = data[2 * numAnchors + a] as number;
    const h = data[3 * numAnchors + a] as number;

    let x1 = cx - w / 2;
    let y1 = cy - h / 2;
    let x2 = cx + w / 2;
    let y2 = cy + h / 2;

    x1 = (x1 - padLeft) / scale;
    y1 = (y1 - padTop) / scale;
    x2 = (x2 - padLeft) / scale;
    y2 = (y2 - padTop) / scale;

    x1 = Math.max(0, Math.min(originalWidth, x1));
    y1 = Math.max(0, Math.min(originalHeight, y1));
    x2 = Math.max(0, Math.min(originalWidth, x2));
    y2 = Math.max(0, Math.min(originalHeight, y2));

    candidates.push({ anchorIdx: a, x1, y1, x2, y2, classId: bestCls, confidence: bestScore });
  }

  if (candidates.length === 0) return emptyDecoded();

  // Build flat arrays then delegate to batchedNms — same algorithm as before
  // but funnelled through the public per-class NMS helper.
  const flatBoxes = new Float32Array(candidates.length * 4);
  const scoresArr = new Float32Array(candidates.length);
  const idxsArr = new Int32Array(candidates.length);
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i] as Candidate;
    flatBoxes[i * 4] = c.x1;
    flatBoxes[i * 4 + 1] = c.y1;
    flatBoxes[i * 4 + 2] = c.x2;
    flatBoxes[i * 4 + 3] = c.y2;
    scoresArr[i] = c.confidence;
    idxsArr[i] = c.classId;
  }
  const kept = batchedNms(flatBoxes, scoresArr, idxsArr, iouThreshold);
  if (kept.length === 0) return emptyDecoded();

  const limited = Array.from(kept).slice(0, maxDetections);
  const k = limited.length;
  const anchorIndices = new Int32Array(k);
  const boxesXyxy = new Float32Array(k * 4);
  const classIds = new Int32Array(k);
  const confidences = new Float32Array(k);
  for (let i = 0; i < k; i++) {
    const c = candidates[limited[i] as number] as Candidate;
    anchorIndices[i] = c.anchorIdx;
    boxesXyxy[i * 4] = c.x1;
    boxesXyxy[i * 4 + 1] = c.y1;
    boxesXyxy[i * 4 + 2] = c.x2;
    boxesXyxy[i * 4 + 3] = c.y2;
    classIds[i] = c.classId;
    confidences[i] = c.confidence;
  }
  return { anchorIndices, boxesXyxy, classIds, confidences };
}

function emptyDecoded(): DecodedAnchors {
  return {
    anchorIndices: new Int32Array(0),
    boxesXyxy: new Float32Array(0),
    classIds: new Int32Array(0),
    confidences: new Float32Array(0),
  };
}

export interface DecodeYoloOptions {
  readonly originalWidth: number;
  readonly originalHeight: number;
  readonly padLeft: number;
  readonly padTop: number;
  readonly scale: number;
  readonly confThreshold: number;
  readonly iouThreshold: number;
  readonly maxDetections: number;
}

export interface DecodedDetection {
  readonly bbox: BoundingBox;
  readonly classId: number;
  readonly confidence: number;
}

/**
 * Decode an anchor-free YOLO detection output into a list of detections.
 *
 * Works for **YOLOv8 / v9 / v10 / v11 / v12** detect heads.
 *
 * Expected raw shape: `[1, 4 + numClasses, N]`. `numClasses` is inferred
 * from the channel count.
 */
export function decodeYolo(
  output: Float32Array,
  outputDims: readonly number[],
  options: DecodeYoloOptions,
): DecodedDetection[] {
  const channels = outputDims.length === 3 ? outputDims[1] : outputDims[0];
  if (channels === undefined || channels < 5) {
    throw new Error(
      `decodeYolo: invalid output channel count ${channels} (expected >= 5).`,
    );
  }
  const numClasses = channels - 4;

  const decoded = decodeYoloAnchors(output, outputDims, {
    numClasses,
    ...options,
  });

  const results: DecodedDetection[] = [];
  for (let i = 0; i < decoded.classIds.length; i++) {
    results.push({
      bbox: new BoundingBox(
        decoded.boxesXyxy[i * 4] as number,
        decoded.boxesXyxy[i * 4 + 1] as number,
        decoded.boxesXyxy[i * 4 + 2] as number,
        decoded.boxesXyxy[i * 4 + 3] as number,
      ),
      classId: decoded.classIds[i] as number,
      confidence: decoded.confidences[i] as number,
    });
  }
  return results;
}

let _warnedDecodeYoloV8 = false;
let _warnedDecodeYoloV8Anchors = false;

/**
 * @deprecated since 0.2.0 — use {@link decodeYolo}. Same behavior; the
 * decoder covers v8/v9/v10/v11/v12 detect heads. Will be removed in 0.3.0.
 */
export function decodeYoloV8(
  output: Float32Array,
  outputDims: readonly number[],
  options: DecodeYoloOptions,
): DecodedDetection[] {
  if (!_warnedDecodeYoloV8) {
    _warnedDecodeYoloV8 = true;
    console.warn(
      "[@ort-vision-sdk/web] decodeYoloV8 is deprecated since 0.2.0; use decodeYolo. " +
        "The alias will be removed in 0.3.0.",
    );
  }
  return decodeYolo(output, outputDims, options);
}

/**
 * @deprecated since 0.2.0 — use {@link decodeYoloAnchors}. Will be removed in 0.3.0.
 */
export function decodeYoloV8Anchors(
  data: Float32Array,
  dims: readonly number[],
  options: DecodeYoloAnchorsOptions,
): DecodedAnchors {
  if (!_warnedDecodeYoloV8Anchors) {
    _warnedDecodeYoloV8Anchors = true;
    console.warn(
      "[@ort-vision-sdk/web] decodeYoloV8Anchors is deprecated since 0.2.0; use decodeYoloAnchors. " +
        "The alias will be removed in 0.3.0.",
    );
  }
  return decodeYoloAnchors(data, dims, options);
}

/** @deprecated since 0.2.0 — use {@link DecodeYoloAnchorsOptions}. */
export type DecodeYoloV8AnchorsOptions = DecodeYoloAnchorsOptions;

/** @deprecated since 0.2.0 — use {@link DecodeYoloOptions}. */
export type DecodeYoloV8Options = DecodeYoloOptions;
