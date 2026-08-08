/**
 * The contract a fused pipeline carries inside its own `.onnx` file.
 *
 * A pipeline built by the Python SDK's `ort_vision_sdk.compose` is a single
 * graph that already contains the detector, the crop-and-resize bridge and the
 * classifier. Everything the runtime needs to drive it — the letterbox
 * resolution, the crop resolution, where the crops come from, how many
 * detections the graph emits, the class names of both stages — was decided at
 * fusion time and written into the model's metadata.
 *
 * This module reads it back. It is the browser half of a contract whose other
 * half lives in `ort_vision_sdk/fusion.py`: the same keys, the same encodings,
 * the same fallbacks. A pipeline fused once therefore runs identically in
 * Python and in a browser tab, off the same file.
 *
 * Building a pipeline stays a Python-side build step — there is no ONNX
 * protobuf writer here, and there is no reason for one: fusing is something you
 * do once next to your export pipeline, not in a page load.
 */

import { parseNames } from "./core/metadata.js";

/**
 * Which tensor the bridge crops the detected boxes out of.
 *
 * - `"detector_input"`: the letterboxed tensor already fed to the detector.
 *   The fused graph then has a **single** image input, but a small object is
 *   cropped out of its downscaled copy.
 * - `"original"`: a second, full-resolution image input. The bridge undoes the
 *   letterbox transform in-graph and crops at native resolution. Two tensors to
 *   feed, still one session and one model load.
 */
export type CropSource = "detector_input" | "original";

/** Value of the `ovs.kind` metadata key for a detector→classifier pipeline. */
export const FUSION_KIND_DETECT_CLASSIFY = "detect_classify";

/**
 * Namespace for every metadata key the fusion writes.
 *
 * Namespaced on purpose: the detector's own Ultralytics metadata (`names`,
 * `task`, `imgsz`) is carried over into the fused model, and an un-prefixed key
 * would either collide with it or be mistaken for it.
 */
export const METADATA_PREFIX = "ovs.";

/** Name of the fused graph's letterboxed detector input, `[1, 3, H, W]` float32 in `[0, 1]`. */
export const INPUT_IMAGE = "images";

/** Name of the full-resolution input. Present only when `cropSource === "original"`. */
export const INPUT_SOURCE = "source_image";

/** Name of the `[1]` float32 letterbox scale factor. Only with `cropSource === "original"`. */
export const INPUT_SCALE = "letterbox_scale";

/** Name of the `[2]` float32 `[padLeft, padTop]`. Only with `cropSource === "original"`. */
export const INPUT_PAD = "letterbox_pad";

/** Name of the `[K, 4]` float32 xyxy output, in **letterboxed** input pixels. */
export const OUTPUT_BOXES = "boxes";

/** Name of the `[K]` float32 detection-confidence output. */
export const OUTPUT_SCORES = "scores";

/** Name of the `[K]` int64 detector-class output. */
export const OUTPUT_CLASSES = "classes";

/** Name of the `[1]` int64 output holding how many of the `K` rows are real. */
export const OUTPUT_NUM_DETECTIONS = "num_detections";

/** Name of the `[K, numClassifierClasses]` float32 classifier output, one row per box. */
export const OUTPUT_PROBS = "probs";

/** Everything a fused pipeline declares about how it must be driven. */
export interface FusionSpec {
  /** Pipeline family. Only `"detect_classify"` exists today. */
  readonly kind: string;
  /** `[width, height]` the detector stage expects — the resolution to letterbox to. */
  readonly inputSize: readonly [number, number];
  /** `[width, height]` every crop is resampled to inside the graph. */
  readonly cropSize: readonly [number, number];
  /** Which tensor the crops are taken from. */
  readonly cropSource: CropSource;
  /**
   * Fixed number of rows `K` every output carries, surplus zero-padded and
   * counted by {@link OUTPUT_NUM_DETECTIONS}. `null` means the graph emits
   * exactly as many rows as survived NMS.
   */
  readonly maxDetections: number | null;
  /** Score threshold baked into the graph's NMS node. */
  readonly confThreshold: number;
  /** IoU threshold baked into the graph's NMS node. */
  readonly iouThreshold: number;
  /** Whether the classifier stage emits logits that still need a softmax. */
  readonly applySoftmax: boolean;
  /** Detector class names in class-id order, or `null` when the fusion recorded none. */
  readonly detectorNames: readonly string[] | null;
  /** Classifier class names in class-id order, or `null`. */
  readonly classifierNames: readonly string[] | null;
  /** Version of `ort-vision-sdk` that produced the file. */
  readonly sdkVersion: string;
  /** Whether driving this pipeline requires feeding the full-resolution input. */
  readonly needsSourceImage: boolean;
}

const KEY_KIND = "kind";
const KEY_SDK_VERSION = "sdk_version";
const KEY_INPUT_SIZE = "input_size";
const KEY_CROP_SIZE = "crop_size";
const KEY_CROP_SOURCE = "crop_source";
const KEY_MAX_DETECTIONS = "max_detections";
const KEY_CONF_THRESHOLD = "conf_threshold";
const KEY_IOU_THRESHOLD = "iou_threshold";
const KEY_APPLY_SOFTMAX = "apply_softmax";
const KEY_DETECTOR_NAMES = "detector_names";
const KEY_CLASSIFIER_NAMES = "classifier_names";

const DYNAMIC = "dynamic";

/**
 * Decode a `"640,640"` pair.
 *
 * @param raw The encoded pair.
 * @returns `[width, height]`, or `null` when the value is missing or malformed.
 */
function decodeSize(raw: string | undefined): readonly [number, number] | null {
  if (!raw) return null;
  const parts = raw.split(",");
  if (parts.length !== 2) return null;
  const width = Number(parts[0]);
  const height = Number(parts[1]);
  if (!Number.isInteger(width) || !Number.isInteger(height)) return null;
  if (width < 1 || height < 1) return null;
  return [width, height];
}

/**
 * Decode a float, falling back when the value is missing or malformed.
 *
 * @param raw The encoded value.
 * @param fallback Value to use when `raw` cannot be read.
 * @returns The parsed number, or `fallback`.
 */
function decodeFloat(raw: string | undefined, fallback: number): number {
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Read a pipeline spec out of a model's custom metadata.
 *
 * Individual malformed entries fall back to the value a fusion would have used
 * by default — a single bad float is not a reason to reject an otherwise
 * loadable pipeline. A malformed resolution is fatal, because there is no safe
 * default for one.
 *
 * @param metadata A model's custom metadata map, as read by
 *   {@link readModelMetadata}.
 * @returns The decoded spec, or `null` when the model is not a fused pipeline —
 *   it carries no `ovs.kind` entry, or one naming a pipeline kind this version
 *   does not know how to drive.
 */
export function readFusionSpec(
  metadata: Readonly<Record<string, string>> | undefined,
): FusionSpec | null {
  if (!metadata) return null;

  const read: Record<string, string> = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (key.startsWith(METADATA_PREFIX)) read[key.slice(METADATA_PREFIX.length)] = value;
  }
  if (read[KEY_KIND] !== FUSION_KIND_DETECT_CLASSIFY) return null;

  const inputSize = decodeSize(read[KEY_INPUT_SIZE]);
  const cropSize = decodeSize(read[KEY_CROP_SIZE]);
  if (inputSize === null || cropSize === null) return null;

  const rawMax = read[KEY_MAX_DETECTIONS] ?? DYNAMIC;
  const parsedMax = Number(rawMax);
  const maxDetections =
    rawMax === DYNAMIC || !Number.isInteger(parsedMax) || parsedMax < 1 ? null : parsedMax;

  const cropSource: CropSource =
    read[KEY_CROP_SOURCE] === "original" ? "original" : "detector_input";

  return {
    kind: FUSION_KIND_DETECT_CLASSIFY,
    inputSize,
    cropSize,
    cropSource,
    maxDetections,
    confThreshold: decodeFloat(read[KEY_CONF_THRESHOLD], 0.25),
    iouThreshold: decodeFloat(read[KEY_IOU_THRESHOLD], 0.45),
    applySoftmax: (read[KEY_APPLY_SOFTMAX] ?? "1") !== "0",
    detectorNames: parseNames(read[KEY_DETECTOR_NAMES]),
    classifierNames: parseNames(read[KEY_CLASSIFIER_NAMES]),
    sdkVersion: read[KEY_SDK_VERSION] ?? "",
    needsSourceImage: cropSource === "original",
  };
}
