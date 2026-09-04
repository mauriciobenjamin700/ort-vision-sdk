/**
 * Which preprocessing a classifier expects its input to have had.
 *
 * A classifier is trained on a specific tensor, and feeding it a differently
 * prepared one degrades it silently — no exception, no warning, just worse
 * predictions. The two families this SDK sees most disagree completely:
 * torchvision-style models want the ImageNet mean and deviation subtracted and
 * divided out, while an Ultralytics classification head consumes raw `[0, 1]`.
 *
 * Guessing wrong is not detectable from the outside, but it is not a guess: an
 * Ultralytics export stamps `author` and `task` into its own metadata, and every
 * task in this SDK already reads that map for the class names.
 *
 * Mirrors `ort_vision_sdk.normalization` in the Python SDK; the two must agree,
 * because the same model file is driven by both.
 */

/**
 * Which preprocessing the classifier expects its input to have had.
 *
 * - `"auto"` (the default wherever it is accepted) reads the model's own export
 *   metadata and picks. An Ultralytics classification head gets
 *   `"ultralytics"`; anything else gets `"imagenet"`.
 * - `"imagenet"` subtracts the ImageNet mean and divides by the ImageNet
 *   deviation — the torchvision convention.
 * - `"ultralytics"` leaves the image in `[0, 1]`. Ultralytics' own classifier
 *   applies no mean/std at all, so anything else feeds it images it never saw
 *   in training.
 * - `"none"` is the same arithmetic as `"ultralytics"` — identity — under a
 *   name that says "this model wants raw `[0, 1]`" rather than naming a vendor.
 */
export type Normalization = "auto" | "imagenet" | "ultralytics" | "none";

/** Per-channel RGB mean of ImageNet, the torchvision preprocessing convention. */
export const IMAGENET_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];

/** Per-channel RGB standard deviation of ImageNet. */
export const IMAGENET_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

/** Mean that leaves an image untouched — what an Ultralytics classifier expects. */
export const IDENTITY_MEAN: readonly [number, number, number] = [0, 0, 0];

/** Deviation that leaves an image untouched — what an Ultralytics classifier expects. */
export const IDENTITY_STD: readonly [number, number, number] = [1, 1, 1];

/** Name reported when the caller supplied `mean`/`std` directly. */
export const CUSTOM_NORMALIZATION = "custom";

const PRESETS: Readonly<
  Record<string, readonly [readonly [number, number, number], readonly [number, number, number]]>
> = {
  imagenet: [IMAGENET_MEAN, IMAGENET_STD],
  ultralytics: [IDENTITY_MEAN, IDENTITY_STD],
  none: [IDENTITY_MEAN, IDENTITY_STD],
};

/** What {@link resolveNormalization} settled on. */
export interface ResolvedNormalization {
  /** The preset name, or `"custom"` when the caller supplied the values. */
  readonly name: string;
  /** Per-channel mean to subtract. */
  readonly mean: readonly [number, number, number];
  /** Per-channel deviation to divide by. */
  readonly std: readonly [number, number, number];
}

/**
 * Whether a metadata map came out of `YOLO(...).export(format="onnx")`.
 *
 * Every Ultralytics export stamps `author` and `task` into its metadata, and the
 * pair is unambiguous: `"Ultralytics"` plus `"classify"` is a classification head
 * from that codebase and nothing else.
 *
 * @param metadata The model's custom metadata map.
 */
export function isUltralyticsClassifier(
  metadata: Readonly<Record<string, string>>,
): boolean {
  return (
    (metadata.author ?? "").trim().toLowerCase() === "ultralytics" &&
    (metadata.task ?? "").trim().toLowerCase() === "classify"
  );
}

/**
 * Settle which `mean`/`std` to apply, and what to call the choice.
 *
 * Explicit `mean`/`std` always win — they are the escape hatch for a model whose
 * preprocessing neither preset describes. Anything they leave open falls back to
 * the preset, so passing only a `mean` does not silently reset the deviation
 * to 1.
 *
 * Warns (via `console.warn`) when the model is an Ultralytics export and the
 * supplied values are not the identity it was trained with. Nothing fails in
 * that case: the prediction has the right shape and is simply less accurate,
 * which is exactly why it is worth saying out loud.
 *
 * @param metadata The model's custom metadata map, read to detect the family.
 * @param options The preset asked for, plus any explicit `mean`/`std`.
 * @throws {RangeError} If `normalization` is not a known preset, or names one
 *   while `mean`/`std` are also supplied — two answers to the same question.
 */
export function resolveNormalization(
  metadata: Readonly<Record<string, string>>,
  options: {
    readonly normalization?: Normalization;
    readonly mean?: readonly [number, number, number];
    readonly std?: readonly [number, number, number];
  } = {},
): ResolvedNormalization {
  const normalization = options.normalization ?? "auto";
  const explicit = options.mean !== undefined || options.std !== undefined;

  if (normalization !== "auto" && PRESETS[normalization] === undefined) {
    throw new RangeError(
      `normalization must be one of 'auto', 'imagenet', 'ultralytics', 'none'; ` +
        `got ${JSON.stringify(normalization)}.`,
    );
  }
  if (explicit && normalization !== "auto") {
    throw new RangeError(
      `Pass either normalization=${JSON.stringify(normalization)} or explicit mean/std, not both.`,
    );
  }

  const ultralytics = isUltralyticsClassifier(metadata);
  const preset =
    normalization === "auto" ? (ultralytics ? "ultralytics" : "imagenet") : normalization;
  const [presetMean, presetStd] = PRESETS[preset] as readonly [
    readonly [number, number, number],
    readonly [number, number, number],
  ];

  if (!explicit) {
    return { name: preset, mean: presetMean, std: presetStd };
  }

  const mean = options.mean ?? presetMean;
  const std = options.std ?? presetStd;
  if (ultralytics && !(isIdentity(mean, IDENTITY_MEAN) && isIdentity(std, IDENTITY_STD))) {
    console.warn(
      "The classifier is an Ultralytics export, whose classification head is trained on raw " +
        `[0, 1] images, but mean=${JSON.stringify(mean)} / std=${JSON.stringify(std)} was ` +
        "requested. It will be fed images normalized in a way it never saw in training, which " +
        "degrades accuracy without raising anything. Drop mean/std to let normalization='auto' " +
        "pick the identity.",
    );
  }
  return { name: CUSTOM_NORMALIZATION, mean, std };
}

/**
 * Whether two channel triples are equal.
 *
 * @param value The triple to check.
 * @param reference The triple to compare against.
 */
function isIdentity(
  value: readonly [number, number, number],
  reference: readonly [number, number, number],
): boolean {
  return value.every((entry, index) => entry === reference[index]);
}
