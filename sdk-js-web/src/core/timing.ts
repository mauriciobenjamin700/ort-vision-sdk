/**
 * Per-stage timing for a single `predict()` call.
 *
 * Populates the `speed` field every `Results` envelope carries, mirroring
 * Ultralytics' `results[0].speed`. All values are milliseconds measured with
 * `performance.now()`.
 */

/**
 * Stage durations of one inference, in milliseconds.
 *
 * `preprocess`, `inference` and `postprocess` are the three keys Ultralytics
 * reports, measured over the same boundaries. `load` is specific to this SDK:
 * `predict()` accepts a URL, `Blob` or DOM element and decodes it internally,
 * so the fetch/decode cost would otherwise be invisible — and on a cold cache
 * it dominates everything else.
 */
export interface Speed {
  /** Fetching and decoding the input into an `RGBImage`. */
  load: number;
  /** Letterbox/resize, normalization and tensor packing. */
  preprocess: number;
  /** The ONNX Runtime forward pass. */
  inference: number;
  /** Decoding raw outputs into results (NMS, mask assembly, top-k). */
  postprocess: number;
}

/**
 * Accumulate stage durations while a `predict()` call runs.
 *
 * Each `stage()` call closes the previous stage: the elapsed time since the
 * last boundary is attributed to the name given. This keeps the call sites
 * free of paired start/stop bookkeeping and guarantees the four stages tile
 * the whole call without gaps.
 */
export class SpeedTimer {
  private _last: number;
  private readonly _speed: Speed = {
    load: 0,
    preprocess: 0,
    inference: 0,
    postprocess: 0,
  };

  constructor() {
    this._last = performance.now();
  }

  /**
   * Attribute the time elapsed since the previous boundary to `stage`.
   *
   * @param stage Which stage just finished.
   */
  stage(stage: keyof Speed): void {
    const now = performance.now();
    this._speed[stage] += now - this._last;
    this._last = now;
  }

  /**
   * The accumulated durations.
   *
   * @returns The `speed` object to hand to the `Results` envelope.
   */
  speed(): Speed {
    return { ...this._speed };
  }
}
