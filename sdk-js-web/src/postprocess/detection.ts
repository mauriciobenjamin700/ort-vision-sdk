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
 * @returns Indices of kept boxes, in descending score order. Boxes tied on
 *   score are visited lowest-index first, so the survivor of a tie is
 *   deterministic and matches both `torchvision` and the Python SDK.
 *
 * The boxes are copied once into score order, so the inner loop walks a
 * contiguous `Float32Array` instead of hopping through `order` into the caller's
 * layout. A pair with no horizontal overlap is skipped before the vertical
 * extent is computed: its IoU is `0`, which suppresses nothing for any
 * non-negative threshold — the shortcut is gated on that so a negative
 * threshold keeps the exact semantics.
 */
export function nms(
  boxes: Float32Array,
  scores: Float32Array,
  iouThreshold: number,
): Int32Array {
  const n = scores.length;
  if (n === 0) return new Int32Array(0);

  const order = new Array<number>(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => (scores[b] as number) - (scores[a] as number) || a - b);

  const sorted = new Float32Array(n * 4);
  const areas = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const i = (order[k] as number) * 4;
    const x1 = boxes[i] as number;
    const y1 = boxes[i + 1] as number;
    const x2 = boxes[i + 2] as number;
    const y2 = boxes[i + 3] as number;
    sorted[k * 4] = x1;
    sorted[k * 4 + 1] = y1;
    sorted[k * 4 + 2] = x2;
    sorted[k * 4 + 3] = y2;
    areas[k] = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  }

  const skipDisjoint = iouThreshold >= 0;
  const suppressed = new Uint8Array(n);
  const keep: number[] = [];

  for (let k = 0; k < n; k++) {
    if (suppressed[k]) continue;
    keep.push(order[k] as number);

    const ax1 = sorted[k * 4] as number;
    const ay1 = sorted[k * 4 + 1] as number;
    const ax2 = sorted[k * 4 + 2] as number;
    const ay2 = sorted[k * 4 + 3] as number;
    const ak = areas[k] as number;

    for (let m = k + 1; m < n; m++) {
      if (suppressed[m]) continue;
      const bx1 = sorted[m * 4] as number;
      const by1 = sorted[m * 4 + 1] as number;
      const bx2 = sorted[m * 4 + 2] as number;
      const by2 = sorted[m * 4 + 3] as number;
      const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
      if (iw === 0 && skipDisjoint) continue;
      const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
      const inter = iw * ih;
      const union = ak + (areas[m] as number) - inter;
      const iou = union > 0 ? inter / union : 0;
      if (iou > iouThreshold) suppressed[m] = 1;
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
 * @returns Indices of kept boxes, sorted by descending score across all
 *   classes. Survivors from different classes that are tied on score are
 *   ordered lowest-index first — an explicit tie-break, because the order the
 *   per-class loop emits them in is an implementation detail (here, `Map`
 *   insertion order; in Python, sorted class order).
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

  keep.sort((a, b) => (scores[b] as number) - (scores[a] as number) || a - b);
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
 * Candidates are found in two passes, mirroring the Python decoder. The first
 * walks each class row contiguously and only *marks* anchors holding some score
 * at or above the threshold — an anchor's best score clears the threshold
 * exactly when one of its scores does, so no running maximum is needed and the
 * hot loop never writes on a typical frame. The anchor-major scan it replaces
 * read `numClasses` values spaced `numAnchors` apart for every one of the 8400
 * anchors and cost ~1 ms even on a frame with no candidates. The winning class
 * is then picked only for the marked anchors — a few hundred, not 8400 — with
 * the same strict `>` that keeps the lowest class id on a tie.
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

  const hit = new Uint8Array(numAnchors);
  const classEnd = (4 + numClasses) * numAnchors;
  for (let row = 4 * numAnchors; row < classEnd; row += numAnchors) {
    for (let a = 0; a < numAnchors; a++) {
      if ((data[row + a] as number) >= confThreshold) hit[a] = 1;
    }
  }

  let count = 0;
  const candidateAnchors = new Int32Array(numAnchors);
  for (let a = 0; a < numAnchors; a++) {
    if (hit[a] === 1) candidateAnchors[count++] = a;
  }
  if (count === 0) return emptyDecoded();

  const flatBoxes = new Float32Array(count * 4);
  const scoresArr = new Float32Array(count);
  const idxsArr = new Int32Array(count);
  for (let i = 0; i < count; i++) {
    const a = candidateAnchors[i] as number;
    const cx = data[a] as number;
    const cy = data[numAnchors + a] as number;
    const w = data[2 * numAnchors + a] as number;
    const h = data[3 * numAnchors + a] as number;
    const x1 = (cx - w / 2 - padLeft) / scale;
    const y1 = (cy - h / 2 - padTop) / scale;
    const x2 = (cx + w / 2 - padLeft) / scale;
    const y2 = (cy + h / 2 - padTop) / scale;
    flatBoxes[i * 4] = Math.max(0, Math.min(originalWidth, x1));
    flatBoxes[i * 4 + 1] = Math.max(0, Math.min(originalHeight, y1));
    flatBoxes[i * 4 + 2] = Math.max(0, Math.min(originalWidth, x2));
    flatBoxes[i * 4 + 3] = Math.max(0, Math.min(originalHeight, y2));
    let bestScore = -Infinity;
    let bestClass = 0;
    for (let c = 0, at = 4 * numAnchors + a; c < numClasses; c++, at += numAnchors) {
      const score = data[at] as number;
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    scoresArr[i] = bestScore;
    idxsArr[i] = bestClass;
  }
  const kept = batchedNms(flatBoxes, scoresArr, idxsArr, iouThreshold);
  if (kept.length === 0) return emptyDecoded();

  const k = Math.min(kept.length, maxDetections);
  const anchorIndices = new Int32Array(k);
  const boxesXyxy = new Float32Array(k * 4);
  const classIds = new Int32Array(k);
  const confidences = new Float32Array(k);
  for (let i = 0; i < k; i++) {
    const j = kept[i] as number;
    anchorIndices[i] = candidateAnchors[j] as number;
    boxesXyxy[i * 4] = flatBoxes[j * 4] as number;
    boxesXyxy[i * 4 + 1] = flatBoxes[j * 4 + 1] as number;
    boxesXyxy[i * 4 + 2] = flatBoxes[j * 4 + 2] as number;
    boxesXyxy[i * 4 + 3] = flatBoxes[j * 4 + 3] as number;
    classIds[i] = idxsArr[j] as number;
    confidences[i] = scoresArr[j] as number;
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
