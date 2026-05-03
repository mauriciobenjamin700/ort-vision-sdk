/**
 * Class label resolution: presets, lists, dicts, or auto-generated.
 *
 * Tasks call {@link resolveLabels} once at construction time to turn whatever
 * the caller passed (preset name, array, dict, or `null`) into an ordered
 * array of class names indexed by class id.
 *
 * In the browser there is no filesystem, so this module does not load labels
 * from a path — fetch the file yourself and pass an array.
 */

import { LabelMapError } from "./core/exceptions.js";

/**
 * Anything accepted by {@link resolveLabels}.
 *
 * - `string[]` / `readonly string[]`: explicit names indexed by class id.
 * - `Record<number, string>`: sparse mapping (gaps filled with `class_<id>`).
 * - `string`: a preset name (e.g. `"coco"`).
 * - `null` / `undefined`: auto-generate `class_0` ... `class_{numClasses-1}`.
 */
export type LabelSpec =
  | readonly string[]
  | Record<number, string>
  | string
  | null
  | undefined;

/** COCO 2017 80-class labels in canonical class-id order. */
export const COCO_CLASSES: readonly string[] = Object.freeze([
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
  "toothbrush",
]);

const PRESETS: Readonly<Record<string, readonly string[]>> = {
  coco: COCO_CLASSES,
};

export interface ResolveLabelsOptions {
  /**
   * Expected number of classes.
   *
   * - When `spec` is `null`/`undefined`, this is required to auto-generate names.
   * - When `spec` is provided, it validates that the resolved length matches.
   */
  readonly numClasses?: number;
}

/**
 * Resolve a labels specification into an ordered array of class names.
 *
 * @throws {@link LabelMapError} if the spec is invalid, the preset is unknown,
 *   or the resolved length disagrees with `numClasses`.
 */
export function resolveLabels(
  spec: LabelSpec,
  options: ResolveLabelsOptions = {},
): readonly string[] {
  const labels = resolve(spec, options.numClasses);
  if (options.numClasses !== undefined && labels.length !== options.numClasses) {
    throw new LabelMapError(
      `Resolved ${labels.length} labels but the model has ${options.numClasses} classes.`,
    );
  }
  return labels;
}

function resolve(spec: LabelSpec, numClasses: number | undefined): readonly string[] {
  if (spec === null || spec === undefined) {
    if (numClasses === undefined) {
      throw new LabelMapError(
        "Cannot auto-generate labels without numClasses. Pass an explicit labels spec or numClasses.",
      );
    }
    return Array.from({ length: numClasses }, (_, i) => `class_${i}`);
  }

  if (Array.isArray(spec)) {
    return [...spec];
  }

  if (typeof spec === "string") {
    const preset = PRESETS[spec];
    if (preset !== undefined) {
      return preset;
    }
    throw new LabelMapError(
      `Unknown labels preset: ${JSON.stringify(spec)}. Known presets: ${Object.keys(PRESETS).join(", ")}.`,
    );
  }

  if (typeof spec === "object") {
    const map = spec as Record<number, string>;
    const ids = Object.keys(map).map((k) => Number(k));
    if (ids.length === 0) {
      return [];
    }
    const maxId = Math.max(...ids);
    return Array.from({ length: maxId + 1 }, (_, i) => map[i] ?? `class_${i}`);
  }

  throw new LabelMapError(`Unsupported labels spec type: ${typeof spec}.`);
}
