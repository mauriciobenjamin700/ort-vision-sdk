/**
 * Common foundation for task-oriented vision SDK objects.
 *
 * The base only owns the {@link OrtSession} — label resolution lives in each
 * subclass because how `numClasses` is read from the model differs per task.
 */

import { NoDetectionsError } from "../core/exceptions.js";
import { OrtSession } from "../core/session.js";
import { type BoundingBox, RGBImage } from "../types.js";

export abstract class VisionTask {
  protected constructor(protected readonly _session: OrtSession) {}

  /** The underlying {@link OrtSession} used to run inference. */
  get session(): OrtSession {
    return this._session;
  }
}

/**
 * Render a confidence threshold the way the Python SDK renders it.
 *
 * JavaScript and Python disagree on how a number becomes text. A whole
 * threshold is `1` here and `1.0` there; JavaScript holds off on exponent
 * notation until `1e-7` while Python switches at `1e-5`. A fused pipeline is
 * built once and runs under both runtimes from the same file, so a message that
 * quotes the threshold has to quote it identically — otherwise the two SDKs
 * describe the same run with two different numbers.
 *
 * Six decimals with the trailing zeros trimmed covers every threshold a caller
 * can meaningfully set and agrees byte for byte with `_format_threshold` in the
 * Python SDK's `tasks/base.py`. The pairing is fixed by a shared table in both
 * test suites, so a change on one side that is not mirrored on the other fails.
 *
 * @param value The threshold to render.
 * @returns The threshold as text, without trailing zeros or a trailing dot.
 */
function formatThreshold(value: number): string {
  return value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

/**
 * Turn an empty result into an error, when the caller asked for that.
 *
 * Shared by every task that can come back with nothing — {@link Detector},
 * {@link Segmenter} and {@link DetectClassify} — so the three agree on when
 * they throw and on what the message says. The message names the two settings
 * that decide the outcome, because "no detections" on its own leaves the reader
 * unable to tell a blank image from a threshold set too high.
 *
 * @param count How many detections survived every filter.
 * @param options The flag for this call, the threshold actually applied (after
 *   any per-call override), the class allowlist if one narrowed the search, and
 *   the source path when the input was one.
 * @throws {@link NoDetectionsError} when the flag is set and `count` is zero.
 */
export function requireDetections(
  count: number,
  options: {
    readonly raiseOnEmpty: boolean;
    readonly confThreshold: number;
    readonly classes: readonly number[] | undefined;
    readonly path: string | null;
  },
): void {
  if (!options.raiseOnEmpty || count > 0) return;
  const where = options.path ? ` in ${options.path}` : "";
  const narrowed =
    options.classes === undefined
      ? ""
      : ` among classes [${[...options.classes].sort((a, b) => a - b).join(", ")}]`;
  throw new NoDetectionsError(
    `No detections${where}${narrowed}: nothing cleared confThreshold=${formatThreshold(options.confThreshold)}.`,
  );
}

/**
 * Cut the box region out of an image.
 *
 * The box is truncated to whole pixels and clamped to the image, the same
 * rounding every task applies, so a crop always matches `bbox.asIntXyxy()`
 * within the frame.
 *
 * @param image The source image.
 * @param bbox The box, in the image's pixel coordinates.
 * @returns The cropped region, or a zero-sized image for a box with no area.
 */
export function cropToBox(image: RGBImage, bbox: BoundingBox): RGBImage {
  const [rawX1, rawY1, rawX2, rawY2] = bbox.asIntXyxy();
  const x1 = Math.max(0, rawX1);
  const y1 = Math.max(0, rawY1);
  const x2 = Math.min(image.width, rawX2);
  const y2 = Math.min(image.height, rawY2);
  if (x2 <= x1 || y2 <= y1) return new RGBImage(new Uint8Array(0), 0, 0);

  const width = x2 - x1;
  const height = y2 - y1;
  const out = new Uint8Array(width * height * 3);
  for (let row = 0; row < height; row++) {
    const offset = ((y1 + row) * image.width + x1) * 3;
    out.set(image.data.subarray(offset, offset + width * 3), row * width * 3);
  }
  return new RGBImage(out, width, height);
}

/**
 * Wrap a computation so it runs on first call and is remembered after.
 *
 * Backs the per-detection images (`croppedImage`, `segmentedImage`): most
 * callers read a box and a class and never look at the pixels, yet building
 * them eagerly copied every box region out of the frame on every
 * `predict()` — up to `maxDetections` allocations nobody asked for.
 *
 * @param compute Produces the value; called at most once.
 * @returns A getter returning the value, computing it on the first call.
 */
export function memoize<T>(compute: () => T): () => T {
  let done = false;
  let value: T;
  return () => {
    if (!done) {
      value = compute();
      done = true;
    }
    return value;
  };
}
