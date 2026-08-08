/**
 * Public output types returned by the SDK's vision tasks.
 *
 * These types form the contract between the SDK and its callers. They mirror
 * the Python `ort-vision-sdk` output dataclasses 1-to-1.
 *
 * Naming is intentionally compatible with the Ultralytics / torchvision idiom
 * (`cls`, `conf`, `box`, `xyxy`, `xywh`, normalized variants) so code ported
 * from those projects keeps working with minimal edits. The original verbose
 * names (`classId`, `className`, `confidence`, `bbox`) are still populated for
 * backwards compatibility.
 */

import { ImageLoadError } from "./core/exceptions.js";

/**
 * HWC RGB uint8 image — the canonical image format used across the SDK.
 *
 * `data.length` must equal `width * height * 3`. The buffer is laid out row
 * by row, top-to-bottom, with each pixel as `[R, G, B]`.
 */
export class RGBImage {
  constructor(
    public readonly data: Uint8Array,
    public readonly width: number,
    public readonly height: number,
  ) {
    if (data.length !== width * height * 3) {
      throw new ImageLoadError(
        `RGBImage data length ${data.length} does not match width * height * 3 = ${
          width * height * 3
        }.`,
      );
    }
  }
}

/**
 * Axis-aligned bounding box in absolute pixel coordinates (xyxy format).
 *
 * Coordinates refer to the original input image (before any internal resize),
 * so callers can map detections back onto their source image without any
 * additional bookkeeping.
 */
export class BoundingBox {
  constructor(
    public readonly x1: number,
    public readonly y1: number,
    public readonly x2: number,
    public readonly y2: number,
  ) {}

  /** Box width in pixels (clamped to non-negative). */
  get width(): number {
    return Math.max(0, this.x2 - this.x1);
  }

  /** Box height in pixels (clamped to non-negative). */
  get height(): number {
    return Math.max(0, this.y2 - this.y1);
  }

  /** Box area in pixels squared. */
  get area(): number {
    return this.width * this.height;
  }

  /** The box as `[x1, y1, x2, y2]` in absolute pixels (Ultralytics-style). */
  get xyxy(): readonly [number, number, number, number] {
    return [this.x1, this.y1, this.x2, this.y2];
  }

  /**
   * The box as `[cx, cy, w, h]` with `(cx, cy)` at the center.
   *
   * Matches Ultralytics' `boxes.xywh` and YOLO's native head format. For the
   * top-left `[x, y, w, h]` convention, use {@link asXywh}.
   */
  get xywh(): readonly [number, number, number, number] {
    return [
      (this.x1 + this.x2) / 2,
      (this.y1 + this.y2) / 2,
      this.width,
      this.height,
    ];
  }

  /**
   * The box as `[x1, y1, x2, y2]` normalized to `[0, 1]`.
   *
   * @param origShape `[height, width]` of the source image, in pixels.
   */
  xyxyn(
    origShape: readonly [number, number],
  ): readonly [number, number, number, number] {
    const [h, w] = origShape;
    if (w <= 0 || h <= 0) return [0, 0, 0, 0];
    return [this.x1 / w, this.y1 / h, this.x2 / w, this.y2 / h];
  }

  /**
   * The box as `[cx, cy, w, h]` normalized to `[0, 1]`.
   *
   * @param origShape `[height, width]` of the source image, in pixels.
   */
  xywhn(
    origShape: readonly [number, number],
  ): readonly [number, number, number, number] {
    const [h, w] = origShape;
    if (w <= 0 || h <= 0) return [0, 0, 0, 0];
    const [cx, cy, bw, bh] = this.xywh;
    return [cx / w, cy / h, bw / w, bh / h];
  }

  /** Returns `[x1, y1, x2, y2]`. */
  asXyxy(): readonly [number, number, number, number] {
    return [this.x1, this.y1, this.x2, this.y2];
  }

  /**
   * Returns `[x, y, width, height]` with `(x, y)` at the **top-left**.
   *
   * Note: this is the top-left convention. Ultralytics' `xywh` getter uses
   * **center** coordinates — for that, read {@link xywh}.
   */
  asXywh(): readonly [number, number, number, number] {
    return [this.x1, this.y1, this.width, this.height];
  }

  /** Returns `[x1, y1, x2, y2]` truncated to integers, useful for slicing arrays. */
  asIntXyxy(): readonly [number, number, number, number] {
    return [
      Math.trunc(this.x1),
      Math.trunc(this.y1),
      Math.trunc(this.x2),
      Math.trunc(this.y2),
    ];
  }
}

/**
 * Probability assigned to a single class for a classification prediction.
 *
 * `cls` / `name` / `conf` are Ultralytics-style aliases populated alongside
 * the verbose `classId` / `className` / `probability` fields.
 */
export interface ClassProbability {
  readonly classId: number;
  readonly className: string;
  readonly probability: number;
  /** Alias for `classId` (Ultralytics-style). */
  readonly cls: number;
  /** Alias for `className`. */
  readonly name: string;
  /** Alias for `probability` (Ultralytics-style). */
  readonly conf: number;
}

/**
 * Output of an image classification inference.
 */
export interface ClassificationResult {
  readonly classId: number;
  readonly className: string;
  readonly confidence: number;
  /** Alias for `classId` (Ultralytics-style). */
  readonly cls: number;
  /** Alias for `className`. */
  readonly name: string;
  /** Alias for `confidence` (Ultralytics-style). */
  readonly conf: number;
  /** The original input image as an HWC RGB uint8 array. */
  readonly image: RGBImage;
  /**
   * Probabilities per class, sorted in descending order. The first entry
   * mirrors `classId`, `className`, and `confidence`. When `topK` was passed
   * to `predict`, the array is truncated to that length.
   */
  readonly probabilities: readonly ClassProbability[];
}

/**
 * Single detected object produced by an object-detection model.
 */
export interface DetectionResult {
  readonly classId: number;
  readonly className: string;
  readonly confidence: number;
  readonly bbox: BoundingBox;
  /** Alias for `classId` (Ultralytics-style). */
  readonly cls: number;
  /** Alias for `className`. */
  readonly name: string;
  /** Alias for `confidence` (Ultralytics-style). */
  readonly conf: number;
  /** Alias for `bbox` (Ultralytics-style). */
  readonly box: BoundingBox;
  /**
   * The original image cropped to `bbox`, HWC RGB uint8. Empty boxes
   * (zero area) yield a zero-sized `RGBImage`.
   */
  readonly croppedImage: RGBImage;
  /**
   * What a second, classification stage predicted **for this crop** —
   * populated only by {@link DetectClassify}, and `null` for a plain detector.
   *
   * Kept as its own field rather than folded into `classId`/`className`
   * because the two answers are different questions: the detector says *what
   * kind of object this is*, the classifier says *which sub-category the object
   * belongs to*, and collapsing them would lose one of the two.
   */
  readonly classification?: ClassificationResult | null;
}

/**
 * Single-channel binary or grayscale mask, laid out row-major.
 *
 * `data.length` must equal `width * height`. For binary masks, values are
 * `0` (background) or `255` (foreground); soft masks may use the full
 * `[0, 255]` range.
 */
export class Mask {
  constructor(
    public readonly data: Uint8Array,
    public readonly width: number,
    public readonly height: number,
  ) {
    if (data.length !== width * height) {
      throw new ImageLoadError(
        `Mask data length ${data.length} does not match width * height = ${
          width * height
        }.`,
      );
    }
  }
}

/**
 * Single segmented instance produced by an instance-segmentation model.
 *
 * Mirrors {@link DetectionResult} and adds the per-instance binary mask
 * plus a "ready-to-display" background-removed crop.
 */
export interface SegmentationResult {
  readonly classId: number;
  readonly className: string;
  readonly confidence: number;
  readonly bbox: BoundingBox;
  /** Alias for `classId` (Ultralytics-style). */
  readonly cls: number;
  /** Alias for `className`. */
  readonly name: string;
  /** Alias for `confidence` (Ultralytics-style). */
  readonly conf: number;
  /** Alias for `bbox` (Ultralytics-style). */
  readonly box: BoundingBox;
  /**
   * Binary mask cropped to `bbox`. Values are `0` (background) or `255`
   * (foreground). Empty boxes yield a zero-sized `Mask`.
   */
  readonly mask: Mask;
  /**
   * The original image cropped to `bbox` with background pixels (where
   * `mask.data[i] === 0`) zeroed out. Empty boxes yield a zero-sized
   * `RGBImage`.
   */
  readonly segmentedImage: RGBImage;
}
