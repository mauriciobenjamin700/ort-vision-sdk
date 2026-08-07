/**
 * Fused letterbox → CHW float32 pipeline with reusable buffers.
 *
 * The composable primitives in {@link ./image.js} each allocate and each walk
 * their input end to end, which is the right shape for a library but the wrong
 * shape for a video loop. Chaining them costs eleven full-buffer passes and six
 * large allocations per frame:
 *
 * `getImageData` → RGBA→RGB → RGB→RGBA → `putImageData` → `drawImage` →
 * `getImageData` → RGBA→RGB → fill → row copies → `toFloat32` → `toCHW`.
 *
 * This module collapses the second half of that into two: one `drawImage` that
 * resizes *and* positions the content inside the padded target in a single
 * accelerated operation, and one loop that reads the resulting RGBA and writes
 * planar float32 directly. The intermediate `RGBImage` at target size, the fill
 * loop, the row copies and the two 4.9 MB `Float32Array` allocations all go
 * away.
 *
 * The primitives stay exactly as they are — they are public API and they are
 * what makes a custom pipeline writable. This is the fast path the built-in
 * tasks take.
 */

import { createCanvas, get2DContext, rgbToImageData, type Canvas2D, type Context2D } from "../core/canvas.js";
import { RGBImage } from "../types.js";

const INV_255 = 1 / 255;

/** Geometry of a letterbox, plus the planar tensor data it produced. */
export interface FusedLetterboxResult {
  /** CHW float32 in `[0, 1]`, length `3 * targetHeight * targetWidth`. */
  readonly data: Float32Array;
  /** Factor applied to the original image (`< 1` if downscaled). */
  readonly scale: number;
  /** Horizontal padding in pixels. */
  readonly padLeft: number;
  /** Vertical padding in pixels. */
  readonly padTop: number;
  /**
   * Whether {@link data} is the pipeline's reusable buffer.
   *
   * `true` means the next {@link LetterboxPipeline.run} overwrites it, so a
   * caller keeping the values past its own inference has to copy them.
   */
  readonly reused: boolean;
}

/**
 * Reusable letterbox → tensor pipeline for one target resolution.
 *
 * Holds a target canvas and an output buffer across calls, so a steady stream
 * of frames at the same size allocates nothing. Create one per task, not per
 * frame.
 */
export class LetterboxPipeline {
  private readonly _targetWidth: number;
  private readonly _targetHeight: number;
  private readonly _fill: readonly [number, number, number];
  private readonly _target: Canvas2D;
  private readonly _targetContext: Context2D;
  private readonly _buffer: Float32Array;
  private _source: Canvas2D | null = null;
  private _sourceContext: Context2D | null = null;
  private _bufferInUse = false;

  /**
   * @param targetWidth Model input width in pixels.
   * @param targetHeight Model input height in pixels.
   * @param fill RGB padding colour; defaults to YOLO grey.
   */
  constructor(
    targetWidth: number,
    targetHeight: number,
    fill: readonly [number, number, number] = [114, 114, 114],
  ) {
    if (targetWidth <= 0 || targetHeight <= 0) {
      throw new Error(`Invalid letterbox target ${targetWidth}x${targetHeight}.`);
    }
    this._targetWidth = targetWidth;
    this._targetHeight = targetHeight;
    this._fill = fill;
    this._target = createCanvas(targetWidth, targetHeight);
    this._targetContext = get2DContext(this._target, { willReadFrequently: true });
    this._targetContext.imageSmoothingEnabled = true;
    this._targetContext.imageSmoothingQuality = "high";
    this._buffer = new Float32Array(3 * targetHeight * targetWidth);
  }

  /** The `[width, height]` this pipeline letterboxes into. */
  get targetSize(): readonly [number, number] {
    return [this._targetWidth, this._targetHeight];
  }

  /**
   * Letterbox an image and write it as planar float32.
   *
   * The returned buffer is reused between calls unless a previous result is
   * still checked out — {@link release} marks it free again. A second `run`
   * before the first is released allocates a fresh buffer rather than
   * corrupting it, so concurrent `predict()` calls on one task stay correct at
   * the cost of the allocation they were trying to avoid.
   *
   * @param image Source image in the SDK's canonical HWC RGB layout.
   */
  run(image: RGBImage): FusedLetterboxResult {
    const targetWidth = this._targetWidth;
    const targetHeight = this._targetHeight;
    const scale = Math.min(targetWidth / image.width, targetHeight / image.height);
    const scaledWidth = Math.round(image.width * scale);
    const scaledHeight = Math.round(image.height * scale);
    const padLeft = Math.floor((targetWidth - scaledWidth) / 2);
    const padTop = Math.floor((targetHeight - scaledHeight) / 2);

    const source = this._ensureSource(image.width, image.height);
    source.putImageData(rgbToImageData(image), 0, 0);

    const context = this._targetContext;
    if (padLeft > 0 || padTop > 0 || scaledWidth !== targetWidth || scaledHeight !== targetHeight) {
      context.fillStyle = `rgb(${this._fill[0]},${this._fill[1]},${this._fill[2]})`;
      context.fillRect(0, 0, targetWidth, targetHeight);
    }
    context.drawImage(
      this._source as CanvasImageSource,
      0,
      0,
      image.width,
      image.height,
      padLeft,
      padTop,
      scaledWidth,
      scaledHeight,
    );

    const rgba = context.getImageData(0, 0, targetWidth, targetHeight).data;
    const reused = !this._bufferInUse;
    const data = reused ? this._buffer : new Float32Array(3 * targetHeight * targetWidth);
    this._bufferInUse = true;

    const plane = targetWidth * targetHeight;
    for (let pixel = 0, offset = 0; pixel < plane; pixel++, offset += 4) {
      data[pixel] = (rgba[offset] as number) * INV_255;
      data[plane + pixel] = (rgba[offset + 1] as number) * INV_255;
      data[2 * plane + pixel] = (rgba[offset + 2] as number) * INV_255;
    }

    return { data, scale, padLeft, padTop, reused };
  }

  /**
   * Mark the reusable buffer free again.
   *
   * Call it once the tensor built from a {@link run} result has been handed to
   * ONNX Runtime and the run has resolved — after that the values are inside
   * the WASM heap and the buffer can be overwritten.
   */
  release(): void {
    this._bufferInUse = false;
  }

  /**
   * Grow the scratch source canvas to fit an image, reusing it when possible.
   *
   * A canvas is only reallocated when a frame arrives at a different size than
   * the last one, which for a camera or video source is never after the first.
   *
   * @param width Source width in pixels.
   * @param height Source height in pixels.
   */
  private _ensureSource(width: number, height: number): Context2D {
    if (this._source === null || this._source.width !== width || this._source.height !== height) {
      this._source = createCanvas(width, height);
      this._sourceContext = get2DContext(this._source);
    }
    return this._sourceContext as Context2D;
  }
}

/**
 * Build a zero-filled CHW tensor payload for a warm-up run.
 *
 * @param width Model input width in pixels.
 * @param height Model input height in pixels.
 */
export function zeroTensorData(width: number, height: number): Float32Array {
  return new Float32Array(3 * height * width);
}

/**
 * Letterbox an image into planar float32 without keeping any state.
 *
 * The allocation-free path is {@link LetterboxPipeline}; this is the one-shot
 * form, for a caller who wants the fused behaviour without owning a pipeline.
 *
 * @param image Source image in the SDK's canonical HWC RGB layout.
 * @param targetWidth Model input width in pixels.
 * @param targetHeight Model input height in pixels.
 * @param fill RGB padding colour; defaults to YOLO grey.
 */
export function letterboxToTensorData(
  image: RGBImage,
  targetWidth: number,
  targetHeight: number,
  fill: readonly [number, number, number] = [114, 114, 114],
): FusedLetterboxResult {
  return new LetterboxPipeline(targetWidth, targetHeight, fill).run(image);
}
