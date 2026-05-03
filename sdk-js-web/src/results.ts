/**
 * Per-image result envelopes — Ultralytics-style `Results` for the SDK.
 *
 * Each `predict()` call returns a 1-element array of these envelopes,
 * mirroring `YOLO("img.jpg")`. Each envelope holds:
 *
 * - A bulk-array view of the predictions (`Boxes`, `Probs`, `Masks`) with
 *   the exact attribute names Ultralytics uses (`xyxy`, `xywh`, `xyxyn`,
 *   `xywhn`, `cls`, `conf`, `data`, `top1`, `top5`).
 * - Per-instance dataclasses (`DetectionResult`, `SegmentationResult`,
 *   `ClassProbability`) for callers who prefer the OO interface.
 * - `names`: `Record<number, string>` matching Ultralytics' `model.names`.
 * - `origImg` / `origShape` / `path`: provenance for the original input.
 */

import {
  type ClassificationResult,
  type DetectionResult,
  type RGBImage,
  type SegmentationResult,
} from "./types.js";

/**
 * Bulk numpy-style view of detected boxes for a single image.
 *
 * Mirrors Ultralytics' `Boxes` interface. Coordinates in {@link xyxy} and
 * {@link xywh} are absolute pixels in the original image; the `*n` variants
 * are normalized to `[0, 1]` using `origShape`.
 */
export class Boxes {
  /**
   * @param xyxy Flat array of length `4 * N` in `[x1, y1, x2, y2, ...]` order.
   * @param cls One class index per box, length `N`.
   * @param conf One confidence per box, length `N`.
   * @param origShape `[height, width]` of the original image.
   */
  constructor(
    public readonly xyxy: Float32Array,
    public readonly cls: Int32Array,
    public readonly conf: Float32Array,
    public readonly origShape: readonly [number, number],
  ) {}

  /** Number of detected boxes. */
  get length(): number {
    return this.cls.length;
  }

  /** `[N, 4]` shape of the `xyxy` view. */
  get shape(): readonly [number, number] {
    return [this.length, 4];
  }

  /** Boxes as `[N, 4]` `[cx, cy, w, h]` flat array in absolute pixels. */
  get xywh(): Float32Array {
    const out = new Float32Array(this.xyxy.length);
    for (let i = 0; i < this.length; i++) {
      const x1 = this.xyxy[i * 4] as number;
      const y1 = this.xyxy[i * 4 + 1] as number;
      const x2 = this.xyxy[i * 4 + 2] as number;
      const y2 = this.xyxy[i * 4 + 3] as number;
      out[i * 4] = (x1 + x2) / 2;
      out[i * 4 + 1] = (y1 + y2) / 2;
      out[i * 4 + 2] = x2 - x1;
      out[i * 4 + 3] = y2 - y1;
    }
    return out;
  }

  /** Boxes as `[N, 4]` `[x1, y1, x2, y2]` normalized to `[0, 1]`. */
  get xyxyn(): Float32Array {
    const [h, w] = this.origShape;
    const out = new Float32Array(this.xyxy.length);
    if (this.length === 0 || w <= 0 || h <= 0) return out;
    for (let i = 0; i < this.length; i++) {
      out[i * 4] = (this.xyxy[i * 4] as number) / w;
      out[i * 4 + 1] = (this.xyxy[i * 4 + 1] as number) / h;
      out[i * 4 + 2] = (this.xyxy[i * 4 + 2] as number) / w;
      out[i * 4 + 3] = (this.xyxy[i * 4 + 3] as number) / h;
    }
    return out;
  }

  /** Boxes as `[N, 4]` `[cx, cy, w, h]` normalized to `[0, 1]`. */
  get xywhn(): Float32Array {
    const xywh = this.xywh;
    const [h, w] = this.origShape;
    if (this.length === 0 || w <= 0 || h <= 0) return xywh;
    for (let i = 0; i < this.length; i++) {
      xywh[i * 4] = (xywh[i * 4] as number) / w;
      xywh[i * 4 + 1] = (xywh[i * 4 + 1] as number) / h;
      xywh[i * 4 + 2] = (xywh[i * 4 + 2] as number) / w;
      xywh[i * 4 + 3] = (xywh[i * 4 + 3] as number) / h;
    }
    return xywh;
  }

  /**
   * Concatenated `[N, 6]` array of `[x1, y1, x2, y2, conf, cls]`.
   *
   * Matches Ultralytics' `boxes.data`.
   */
  get data(): Float32Array {
    const out = new Float32Array(this.length * 6);
    for (let i = 0; i < this.length; i++) {
      out[i * 6] = this.xyxy[i * 4] as number;
      out[i * 6 + 1] = this.xyxy[i * 4 + 1] as number;
      out[i * 6 + 2] = this.xyxy[i * 4 + 2] as number;
      out[i * 6 + 3] = this.xyxy[i * 4 + 3] as number;
      out[i * 6 + 4] = this.conf[i] as number;
      out[i * 6 + 5] = this.cls[i] as number;
    }
    return out;
  }
}

/**
 * Top-k classification probabilities for a single image.
 *
 * Mirrors Ultralytics' `Probs` interface.
 */
export class Probs {
  /** @param data `[numClasses]` per-class probabilities, indexed by class id. */
  constructor(public readonly data: Float32Array) {}

  /** Number of classes. */
  get length(): number {
    return this.data.length;
  }

  /** `[numClasses]` shape of the underlying vector. */
  get shape(): readonly [number] {
    return [this.length];
  }

  /** Index of the most probable class. */
  get top1(): number {
    if (this.data.length === 0) return 0;
    let best = 0;
    let bestVal = this.data[0] as number;
    for (let i = 1; i < this.data.length; i++) {
      const v = this.data[i] as number;
      if (v > bestVal) {
        best = i;
        bestVal = v;
      }
    }
    return best;
  }

  /** Probability of the top-1 class. */
  get top1conf(): number {
    if (this.data.length === 0) return 0;
    return this.data[this.top1] as number;
  }

  /** Indices of the top-5 most probable classes, descending. */
  get top5(): Int32Array {
    return this._topK(5).indices;
  }

  /** Probabilities of the top-5 classes, descending. */
  get top5conf(): Float32Array {
    return this._topK(5).values;
  }

  private _topK(k: number): { indices: Int32Array; values: Float32Array } {
    const n = Math.min(k, this.data.length);
    const order: number[] = [];
    for (let i = 0; i < this.data.length; i++) order.push(i);
    order.sort((a, b) => (this.data[b] as number) - (this.data[a] as number));
    const indices = new Int32Array(n);
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      indices[i] = order[i] as number;
      values[i] = this.data[order[i] as number] as number;
    }
    return { indices, values };
  }
}

/**
 * Per-instance binary masks for a single image.
 *
 * Each mask is cropped to its instance's bounding box. To paint masks onto
 * a full-image canvas, use `xyxy[i]` as the top-left target.
 */
export class Masks {
  /**
   * @param data Per-instance binary masks (`Mask` objects from `types.ts`).
   * @param xyxy Flat `[N, 4]` of bounding-box coordinates in original pixels.
   * @param origShape `[height, width]` of the original image.
   */
  constructor(
    public readonly data: ReadonlyArray<{
      readonly data: Uint8Array;
      readonly width: number;
      readonly height: number;
    }>,
    public readonly xyxy: Float32Array,
    public readonly origShape: readonly [number, number],
  ) {}

  /** Number of instance masks. */
  get length(): number {
    return this.data.length;
  }

  /** `[N]` shape of the masks collection. */
  get shape(): readonly [number] {
    return [this.length];
  }

  [Symbol.iterator](): Iterator<{
    readonly data: Uint8Array;
    readonly width: number;
    readonly height: number;
  }> {
    return this.data[Symbol.iterator]();
  }
}

/**
 * Per-image detection envelope (Ultralytics-style `Results`).
 *
 * Iterating yields per-instance {@link DetectionResult} entries, so legacy
 * code that did `for (const d of detector.predict(img))` only needs an
 * extra `[0]` to bridge:
 *
 * ```typescript
 * for (const d of (await detector.predict(img))[0]) {
 *   console.log(d.cls, d.conf, d.box.xyxy);
 * }
 * ```
 *
 * For numpy-style bulk access, use the `boxes` collection.
 */
export class DetectionResults implements Iterable<DetectionResult> {
  constructor(
    public readonly boxes: Boxes,
    public readonly detections: readonly DetectionResult[],
    public readonly names: Readonly<Record<number, string>>,
    public readonly origImg: RGBImage,
    public readonly origShape: readonly [number, number],
    public readonly path: string | null = null,
    public readonly speed: Readonly<Record<string, number>> = {},
  ) {}

  /** Number of surviving detections. */
  get length(): number {
    return this.detections.length;
  }

  /** Index into the per-instance detections. */
  get(index: number): DetectionResult | undefined {
    return this.detections[index];
  }

  [Symbol.iterator](): Iterator<DetectionResult> {
    return this.detections[Symbol.iterator]();
  }
}

/**
 * Per-image classification envelope (Ultralytics-style `Results`).
 */
export class ClassificationResults {
  constructor(
    public readonly probs: Probs,
    public readonly result: ClassificationResult,
    public readonly names: Readonly<Record<number, string>>,
    public readonly origImg: RGBImage,
    public readonly origShape: readonly [number, number],
    public readonly path: string | null = null,
    public readonly speed: Readonly<Record<string, number>> = {},
  ) {}

  /** Top-1 class index (Ultralytics-style alias). */
  get cls(): number {
    return this.probs.top1;
  }

  /** Top-1 confidence (Ultralytics-style alias). */
  get conf(): number {
    return this.probs.top1conf;
  }

  /** Top-1 class name. */
  get name(): string {
    return this.names[this.cls] ?? `class_${this.cls}`;
  }

  /** Per-class probability list, sorted descending (legacy field). */
  get probabilities(): readonly ClassificationResult["probabilities"][number][] {
    return this.result.probabilities;
  }
}

/**
 * Per-image instance-segmentation envelope (Ultralytics-style `Results`).
 *
 * Iterating yields per-instance {@link SegmentationResult} entries. `boxes`
 * and `masks` mirror Ultralytics' bulk-array views.
 */
export class SegmentationResults implements Iterable<SegmentationResult> {
  constructor(
    public readonly boxes: Boxes,
    public readonly masks: Masks,
    public readonly detections: readonly SegmentationResult[],
    public readonly names: Readonly<Record<number, string>>,
    public readonly origImg: RGBImage,
    public readonly origShape: readonly [number, number],
    public readonly path: string | null = null,
    public readonly speed: Readonly<Record<string, number>> = {},
  ) {}

  /** Number of surviving instances. */
  get length(): number {
    return this.detections.length;
  }

  /** Index into the per-instance results. */
  get(index: number): SegmentationResult | undefined {
    return this.detections[index];
  }

  [Symbol.iterator](): Iterator<SegmentationResult> {
    return this.detections[Symbol.iterator]();
  }
}
