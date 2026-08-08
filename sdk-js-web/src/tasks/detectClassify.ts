/**
 * Run a fused detect→classify pipeline in the browser.
 *
 * The file this loads was built by the Python SDK's `ort_vision_sdk.compose`,
 * and it already contains both models plus the crop-and-resize bridge between
 * them. That matters far more in a tab than on a server: two models mean two
 * `.onnx` downloads, two WASM/WebGPU session initializations, and a per-crop
 * round trip through JavaScript to slice, resize and restack the regions before
 * the second model can see them. A fused pipeline has one download, one session
 * and no round trip — the crops are produced and consumed inside the graph.
 */

import type * as ort from "onnxruntime-web";

import { FusionError } from "../core/exceptions.js";
import { type ModelSource, type OrtSessionOptions, OrtSession } from "../core/session.js";
import { SpeedTimer } from "../core/timing.js";
import {
  INPUT_IMAGE,
  INPUT_PAD,
  INPUT_SCALE,
  INPUT_SOURCE,
  OUTPUT_BOXES,
  OUTPUT_CLASSES,
  OUTPUT_NUM_DETECTIONS,
  OUTPUT_PROBS,
  OUTPUT_SCORES,
  type FusionSpec,
  readFusionSpec,
} from "../fusion.js";
import { type ImageInput, loadImage } from "../io/image.js";
import { type LabelSpec, resolveLabels } from "../labels.js";
import { softmax, topK } from "../postprocess/classification.js";
import { toCHW, toFloat32, toFloat32Tensor } from "../preprocess/image.js";
import { LetterboxPipeline, zeroTensorData } from "../preprocess/pipeline.js";
import { Boxes, DetectClassifyResults } from "../results.js";
import {
  BoundingBox,
  RGBImage,
  type ClassProbability,
  type ClassificationResult,
  type DetectionResult,
} from "../types.js";
import { VisionTask, requireDetections } from "./base.js";

export interface DetectClassifyOptions extends OrtSessionOptions {
  /**
   * Class label spec for the **detection** stage — see {@link resolveLabels}.
   * Defaults to the names recorded at fusion time, falling back to the COCO
   * 80-class preset when the fusion recorded none.
   */
  readonly labels?: LabelSpec;
  /**
   * Class label spec for the **classification** stage. Defaults to the recorded
   * names, falling back to generated `class_<id>` names.
   */
  readonly classifierLabels?: LabelSpec;
  /**
   * If `true`, a run that finds nothing throws {@link NoDetectionsError}
   * instead of returning an empty envelope. Default `false`, because looking
   * and finding nothing is a successful inference. Turn it on when an empty
   * result means the surrounding pipeline should stop rather than carry on with
   * zero rows. Can be overridden per `predict` call.
   */
  readonly raiseOnEmpty?: boolean;
}

export interface DetectClassifyPredictOptions {
  /**
   * Drop detections scoring below this. The graph's own NMS threshold was fixed
   * at fusion time and cannot be lowered here — this only filters further.
   */
  readonly confThreshold?: number;
  /** If set, keep only detections whose detector `classId` is in this list. */
  readonly classes?: readonly number[];
  /** Truncate each detection's `classification.probabilities` to its top-k entries. */
  readonly topK?: number;
  /** Override the constructor's `raiseOnEmpty` setting for this call. */
  readonly raiseOnEmpty?: boolean;
}

/**
 * Detector and classifier running as a single ONNX model.
 *
 * Everything the pipeline needs to know about itself — the resolution to
 * letterbox to, whether it wants the full-resolution image as well, whether its
 * classifier output still needs a softmax, the class names of both stages — was
 * written into the file at fusion time and is read back here. Nothing is
 * restated on the JavaScript side, so nothing can drift out of step with the
 * Python side that built it.
 *
 * @example
 * ```typescript
 * const pipeline = await DetectClassify.create("/models/pipeline.onnx");
 * const result = (await pipeline.predict("/images/flock.jpg"))[0];
 * for (const detection of result) {
 *   console.log(detection.name, detection.conf, detection.classification?.name);
 * }
 * ```
 */
export class DetectClassify extends VisionTask {
  private constructor(
    session: OrtSession,
    private readonly _spec: FusionSpec,
    private readonly _labels: readonly string[],
    private readonly _names: Readonly<Record<number, string>>,
    private readonly _classifierLabels: readonly string[],
    private readonly _classifierNames: Readonly<Record<number, string>>,
    private readonly _raiseOnEmpty: boolean,
  ) {
    super(session);
  }

  private _pipelineCache: LetterboxPipeline | null = null;

  /**
   * Run the model once on zero-filled inputs, paying one-time costs up front.
   *
   * Worth more here than on a single-stage task: a fused pipeline is two models
   * plus the bridge in one graph, so the first inference compiles shaders for
   * all of it. Calling this while a loading spinner is still up moves that cost
   * somewhere the user is already waiting.
   *
   * @param runs How many warm-up inferences to run. One is enough for WASM;
   *   WebGPU sometimes settles on the second.
   */
  async warmup(runs: number = 1): Promise<void> {
    const [width, height] = this._spec.inputSize;
    for (let i = 0; i < runs; i++) {
      const feeds: Record<string, ort.Tensor> = {
        [INPUT_IMAGE]: toFloat32Tensor(zeroTensorData(width, height), [1, 3, height, width]),
      };
      if (this._spec.needsSourceImage) {
        feeds[INPUT_SOURCE] = toFloat32Tensor(
          zeroTensorData(width, height),
          [1, 3, height, width],
        );
        feeds[INPUT_SCALE] = toFloat32Tensor(new Float32Array([1]), [1]);
        feeds[INPUT_PAD] = toFloat32Tensor(new Float32Array([0, 0]), [2]);
      }
      await this._session.run(feeds);
    }
  }

  /**
   * The fused preprocessing pipeline, built on first use.
   *
   * Lazily, because constructing it allocates canvases: a pipeline built in an
   * environment without a canvas implementation stays constructible, and only
   * fails if it is actually asked to preprocess something.
   */
  private get _pipeline(): LetterboxPipeline {
    if (this._pipelineCache === null) {
      const [width, height] = this._spec.inputSize;
      this._pipelineCache = new LetterboxPipeline(width, height);
    }
    return this._pipelineCache;
  }

  /**
   * Load a fused pipeline and resolve both label spaces.
   *
   * @param model The fused `.onnx` — a URL, an `ArrayBuffer`, or bytes.
   * @param options Label overrides plus the usual session options.
   * @throws {@link FusionError} when the model carries no pipeline metadata,
   *   i.e. it is a plain detector or classifier rather than something
   *   `ort_vision_sdk.compose` produced.
   */
  static async create(
    model: ModelSource,
    options: DetectClassifyOptions = {},
  ): Promise<DetectClassify> {
    const session = await OrtSession.create(model, options);
    const spec = readFusionSpec(session.metadata);
    if (spec === null) {
      throw new FusionError(
        "This model carries no fused-pipeline metadata, so DetectClassify cannot tell how to " +
          "drive it. Build one with ort_vision_sdk.compose.fuse_detect_classify, or load a " +
          "plain model with Detector/Classifier instead.",
      );
    }

    const labels = resolveLabels(options.labels ?? spec.detectorNames ?? "coco");
    const classifierLabels = resolveLabels(options.classifierLabels ?? spec.classifierNames, {
      numClasses: classifierClasses(session) ?? undefined,
    });
    return new DetectClassify(
      session,
      spec,
      labels,
      indexNames(labels),
      classifierLabels,
      indexNames(classifierLabels),
      options.raiseOnEmpty ?? false,
    );
  }

  /** The pipeline configuration recorded in the model at fusion time. */
  get spec(): FusionSpec {
    return this._spec;
  }

  /** The `[width, height]` the detection stage runs at. */
  get inputSize(): readonly [number, number] {
    return this._spec.inputSize;
  }

  /** Detector class labels indexed by class id. */
  get labels(): readonly string[] {
    return this._labels;
  }

  /** Detector class id → class name (matches Ultralytics' `model.names`). */
  get names(): Readonly<Record<number, string>> {
    return this._names;
  }

  /** Classifier class labels indexed by class id. */
  get classifierLabels(): readonly string[] {
    return this._classifierLabels;
  }

  /** Classifier class id → class name. */
  get classifierNames(): Readonly<Record<number, string>> {
    return this._classifierNames;
  }

  /**
   * Alias for {@link predict} — call the pipeline like a torch `nn.Module`.
   *
   * Use as `pipeline.call(img)` since JavaScript class instances are not
   * callable; for direct invocation, prefer `pipeline.predict(img)`.
   */
  async call(
    image: ImageInput,
    options: DetectClassifyPredictOptions = {},
  ): Promise<DetectClassifyResults[]> {
    return this.predict(image, options);
  }

  /**
   * Run the pipeline on a single image.
   *
   * The returned envelope carries a {@link Speed} breakdown in `speed`. Its
   * `inference` figure covers detection *and* classification, since the
   * pipeline runs them as one graph and no boundary between them is observable
   * from outside.
   */
  async predict(
    image: ImageInput,
    options: DetectClassifyPredictOptions = {},
  ): Promise<DetectClassifyResults[]> {
    const timer = new SpeedTimer();
    const path = typeof image === "string" ? image : null;
    const original = await loadImage(image);
    timer.stage("load");
    const { feeds, scale, padLeft, padTop } = this._preprocess(original);
    timer.stage("preprocess");
    const outputs = await this._session.run(feeds);
    this._pipeline.release();
    timer.stage("inference");

    const probsTensor = output(outputs, OUTPUT_PROBS);
    const boxes = floats(outputs, OUTPUT_BOXES);
    const scores = floats(outputs, OUTPUT_SCORES);
    const classes = integers(outputs, OUTPUT_CLASSES);
    const probs = probsTensor.data as Float32Array;
    const reported = integers(outputs, OUTPUT_NUM_DETECTIONS)[0] ?? 0;
    const rows = Math.min(reported, Math.floor(boxes.length / 4));
    const classCount = probsTensor.dims[probsTensor.dims.length - 1] ?? 0;

    const allowed = options.classes === undefined ? null : new Set(options.classes);
    const floor = options.confThreshold ?? 0;
    const detections: DetectionResult[] = [];
    for (let row = 0; row < rows; row++) {
      const classId = classes[row] ?? 0;
      const confidence = scores[row] ?? 0;
      if (confidence < floor || (allowed !== null && !allowed.has(classId))) continue;

      const bbox = this._toOriginal(boxes, row, { scale, padLeft, padTop, original });
      const cropped = crop(original, bbox);
      detections.push(
        detection(
          classId,
          this._names[classId] ?? `class_${classId}`,
          confidence,
          bbox,
          cropped,
          this._classify(probs.subarray(row * classCount, (row + 1) * classCount), cropped, options.topK),
        ),
      );
    }

    requireDetections(detections.length, {
      raiseOnEmpty: options.raiseOnEmpty ?? this._raiseOnEmpty,
      confThreshold: Math.max(floor, this._spec.confThreshold),
      classes: options.classes,
      path,
    });

    const origShape: readonly [number, number] = [original.height, original.width];
    timer.stage("postprocess");
    return [
      new DetectClassifyResults(
        bulkBoxes(detections, origShape),
        detections,
        this._names,
        this._classifierNames,
        original,
        origShape,
        path,
        timer.speed(),
      ),
    ];
  }

  /**
   * Letterbox the image and build the graph's feeds.
   *
   * The detector input runs through {@link LetterboxPipeline}, which fuses the
   * resize, the padding and the HWC-to-CHW float conversion into one
   * `drawImage` plus one readback, and reuses its output buffer between frames.
   * That buffer goes straight to ONNX Runtime, so `_pipeline.release()` must not
   * be called until the run resolves.
   *
   * A pipeline fused with `cropSource: "original"` also takes the untouched
   * image as a second input, plus the scale and padding of the letterbox — that
   * is what lets the graph undo the letterbox transform internally and crop at
   * native resolution instead of from the downscaled copy. That one is **not**
   * letterboxed by definition, so it does not go through the fused path.
   */
  private _preprocess(image: RGBImage): {
    feeds: Record<string, ort.Tensor>;
    scale: number;
    padLeft: number;
    padTop: number;
  } {
    const [width, height] = this._spec.inputSize;
    const boxed = this._pipeline.run(image);
    const feeds: Record<string, ort.Tensor> = {
      [INPUT_IMAGE]: toFloat32Tensor(boxed.data, [1, 3, height, width]),
    };

    if (this._spec.needsSourceImage) {
      feeds[INPUT_SOURCE] = tensorOf(image);
      feeds[INPUT_SCALE] = toFloat32Tensor(new Float32Array([boxed.scale]), [1]);
      feeds[INPUT_PAD] = toFloat32Tensor(
        new Float32Array([boxed.padLeft, boxed.padTop]),
        [2],
      );
    }
    return { feeds, scale: boxed.scale, padLeft: boxed.padLeft, padTop: boxed.padTop };
  }

  /**
   * Map one letterboxed xyxy row back onto the original image.
   *
   * The graph always reports boxes in the detector's letterboxed pixel space,
   * whichever crop source it was fused with, so both sources agree here.
   */
  private _toOriginal(
    boxes: Float32Array,
    row: number,
    context: {
      scale: number;
      padLeft: number;
      padTop: number;
      original: RGBImage;
    },
  ): BoundingBox {
    const { scale, padLeft, padTop, original } = context;
    const at = (offset: number): number => boxes[row * 4 + offset] ?? 0;
    const clampX = (value: number): number => Math.min(Math.max(value, 0), original.width);
    const clampY = (value: number): number => Math.min(Math.max(value, 0), original.height);
    return new BoundingBox(
      clampX((at(0) - padLeft) / scale),
      clampY((at(1) - padTop) / scale),
      clampX((at(2) - padLeft) / scale),
      clampY((at(3) - padTop) / scale),
    );
  }

  /**
   * Turn one row of the classifier output into a result object.
   *
   * @param row The output row for this detection.
   * @param image The crop the row describes, carried so callers can display
   *   what was classified.
   * @param k Optional truncation of the probability list.
   */
  private _classify(
    row: Float32Array,
    image: RGBImage,
    k: number | undefined,
  ): ClassificationResult {
    const scores = this._spec.applySoftmax ? softmax(row) : row;
    const { indices, values } = topK(scores, k ?? null);
    const probabilities: ClassProbability[] = [];
    for (let i = 0; i < indices.length; i++) {
      const classId = indices[i] ?? 0;
      const probability = values[i] ?? 0;
      const className = this._classifierLabels[classId] ?? `class_${classId}`;
      probabilities.push({
        classId,
        className,
        probability,
        cls: classId,
        name: className,
        conf: probability,
      });
    }
    const top = probabilities[0] ?? {
      classId: 0,
      className: "class_0",
      probability: 0,
      cls: 0,
      name: "class_0",
      conf: 0,
    };
    return {
      classId: top.classId,
      className: top.className,
      confidence: top.probability,
      cls: top.classId,
      name: top.className,
      conf: top.probability,
      image,
      probabilities,
    };
  }
}

/**
 * Read the classifier stage's class count off the `probs` output shape.
 *
 * @param session The loaded pipeline session.
 * @returns The class count, or `null` when the graph leaves that axis dynamic
 *   or declares no `probs` output — in which case label resolution falls back
 *   to whatever the fusion recorded.
 */
function classifierClasses(session: OrtSession): number | null {
  const index = session.outputNames.indexOf(OUTPUT_PROBS);
  if (index < 0) return null;
  const shape = session.outputShapes[index];
  if (shape === undefined || shape.length === 0) return null;
  return shape[shape.length - 1] ?? null;
}

/**
 * Build a class id → name record from an ordered label list.
 *
 * @param labels Labels indexed by class id.
 * @returns The equivalent record.
 */
function indexNames(labels: readonly string[]): Readonly<Record<number, string>> {
  const names: Record<number, string> = {};
  for (let i = 0; i < labels.length; i++) names[i] = labels[i] as string;
  return names;
}

/**
 * Fetch an output tensor by name.
 *
 * @param outputs The run's results.
 * @param name The output's name in the pipeline contract.
 * @returns The tensor.
 * @throws {@link FusionError} when the graph does not carry that output, which
 *   means the file is not a pipeline this version can drive.
 */
function output(outputs: Record<string, ort.Tensor>, name: string): ort.Tensor {
  const tensor = outputs[name];
  if (tensor === undefined) {
    throw new FusionError(`The fused pipeline is missing its '${name}' output.`);
  }
  return tensor;
}

/**
 * Fetch a float output by name.
 *
 * @param outputs The run's results.
 * @param name The output's name in the pipeline contract.
 * @returns The output's data.
 * @throws {@link FusionError} when the graph does not carry that output.
 */
function floats(outputs: Record<string, ort.Tensor>, name: string): Float32Array {
  return output(outputs, name).data as Float32Array;
}

/**
 * Fetch an integer output by name, normalizing ORT's 64-bit representation.
 *
 * ONNX Runtime Web returns `int64` tensors as `BigInt64Array`, whose values do
 * not compare or index like numbers. Class ids and detection counts are always
 * small, so widening them to `number` here is lossless and keeps every caller
 * free of `BigInt` handling.
 *
 * @param outputs The run's results.
 * @param name The output's name in the pipeline contract.
 * @returns The output's values as plain numbers.
 * @throws {@link FusionError} when the graph does not carry that output.
 */
function integers(outputs: Record<string, ort.Tensor>, name: string): number[] {
  const data = output(outputs, name).data as BigInt64Array | Int32Array | Float32Array;
  const values: number[] = [];
  for (let i = 0; i < data.length; i++) values.push(Number(data[i]));
  return values;
}

/**
 * Convert an image to the `[1, 3, H, W]` float32 tensor the graph expects.
 *
 * @param image The image to convert.
 * @returns The batched CHW tensor, scaled to `[0, 1]`.
 */
function tensorOf(image: RGBImage): ort.Tensor {
  const chw = toCHW(toFloat32(image), image.width, image.height, 3);
  return toFloat32Tensor(chw, [1, 3, image.height, image.width]);
}

/**
 * Cut the box region out of the original image.
 *
 * @param image The source image.
 * @param bbox The box, in original-image pixel coordinates.
 * @returns The cropped region, or a zero-sized image for a box with no area.
 */
function crop(image: RGBImage, bbox: BoundingBox): RGBImage {
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
 * Assemble one detection, filling the Ultralytics-style aliases.
 *
 * @param classId Detector class index.
 * @param className Detector class name.
 * @param confidence Detection score.
 * @param bbox Box in original-image coordinates.
 * @param croppedImage The region the box covers.
 * @param classification What the classification stage said about that region.
 * @returns The detection object.
 */
function detection(
  classId: number,
  className: string,
  confidence: number,
  bbox: BoundingBox,
  croppedImage: RGBImage,
  classification: ClassificationResult,
): DetectionResult {
  return {
    classId,
    className,
    confidence,
    bbox,
    cls: classId,
    name: className,
    conf: confidence,
    box: bbox,
    croppedImage,
    classification,
  };
}

/**
 * Assemble the bulk-array `Boxes` view from per-instance detections.
 *
 * @param detections The surviving detections.
 * @param origShape `[height, width]` of the original image.
 * @returns The bulk view, empty when nothing survived.
 */
function bulkBoxes(
  detections: readonly DetectionResult[],
  origShape: readonly [number, number],
): Boxes {
  const xyxy = new Float32Array(detections.length * 4);
  const cls = new Int32Array(detections.length);
  const conf = new Float32Array(detections.length);
  detections.forEach((entry, index) => {
    const [x1, y1, x2, y2] = entry.bbox.xyxy;
    xyxy[index * 4] = x1;
    xyxy[index * 4 + 1] = y1;
    xyxy[index * 4 + 2] = x2;
    xyxy[index * 4 + 3] = y2;
    cls[index] = entry.classId;
    conf[index] = entry.confidence;
  });
  return new Boxes(xyxy, cls, conf, origShape);
}
