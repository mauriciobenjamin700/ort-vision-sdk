/**
 * Object detection task using anchor-free YOLO ONNX models (v8/v9/v10/v11/v12).
 */

import type * as ort from "onnxruntime-web";

import { type ModelSource, type OrtSessionOptions, OrtSession } from "../core/session.js";
import { SpeedTimer } from "../core/timing.js";

import { type ImageInput, loadImage } from "../io/image.js";
import { detectionNumClasses, resolveInputSize } from "../core/graph.js";
import { modelNames } from "../core/metadata.js";
import { type LabelSpec, resolveLabels } from "../labels.js";
import { decodeYolo } from "../postprocess/detection.js";
import { toFloat32Tensor } from "../preprocess/image.js";
import { LetterboxPipeline, zeroTensorData } from "../preprocess/pipeline.js";
import { Boxes, DetectionResults } from "../results.js";
import { VisionTask } from "./base.js";
import {
  type BoundingBox,
  type DetectionResult,
  RGBImage,
} from "../types.js";

/**
 * Decoder family for the model's detection head.
 *
 * - `"yolo"`: anchor-free YOLO head with output shape `[1, 4 + nc, N]` —
 *   covers YOLOv8, v9, v10, v11, v12, v26 detect exports.
 *
 * The SDK does **not** auto-detect the head from the model — the caller is
 * responsible for picking a head that matches their export. Future families
 * (v5/v6/v7 with `[1, N, 5+nc]`) will be added as new literal members.
 */
export type DetectorHead = "yolo";

export interface DetectorOptions extends OrtSessionOptions {
  /**
   * Decoder family for the detection head. Default `"yolo"` covers
   * YOLOv8/v9/v10/v11/v12/v26.
   */
  readonly head?: DetectorHead;
  /** Class label spec — see {@link resolveLabels}. Defaults to the COCO 80-class preset. */
  readonly labels?: LabelSpec;
  /** Number of classes — used to validate the supplied labels. */
  readonly numClasses?: number;
  /**
   * Model input `[width, height]` in pixels for letterboxing.
   *
   * Only used when the model's graph leaves its spatial axes dynamic: a graph
   * that declares a static size always wins, since that is the only shape ONNX
   * Runtime will accept. Defaults to `[640, 640]`.
   */
  readonly inputSize?: readonly [number, number];
  /** Default minimum class score to keep a candidate. */
  readonly confThreshold?: number;
  /** Default IoU threshold for non-maximum suppression. */
  readonly iouThreshold?: number;
  /** Maximum number of detections per image. */
  readonly maxDetections?: number;
}

export interface DetectorPredictOptions {
  /** Override the default confidence threshold. */
  readonly confThreshold?: number;
  /** Override the default IoU threshold. */
  readonly iouThreshold?: number;
  /**
   * If set, keep only detections whose `classId` is in this list.
   * Mirrors Ultralytics' `model.predict(img, classes=[0, 16])`.
   */
  readonly classes?: readonly number[];
}

/**
 * Object detector for anchor-free YOLO ONNX models (v8/v9/v10/v11/v12).
 *
 * `predict()` returns `Promise<DetectionResults[]>` (length 1 for a single
 * image), mirroring Ultralytics' `YOLO("img.jpg")`. Iterate the envelope for
 * per-instance dataclasses, or use the bulk `boxes` view (`.xyxy`, `.xywh`,
 * `.xyxyn`, `.xywhn`, `.cls`, `.conf`).
 *
 * @example
 * ```typescript
 * const det = await Detector.create("/models/yolov8n.onnx");
 * const results = await det.predict("/images/street.jpg");
 * const r = results[0];
 * console.log(r.boxes.xyxy, r.boxes.cls, r.boxes.conf, r.names);
 * for (const d of r) {
 *   console.log(d.cls, d.conf, d.box.xyxy);
 * }
 * ```
 */
export class Detector extends VisionTask {
  private constructor(
    session: OrtSession,
    private readonly _head: DetectorHead,
    private readonly _labels: readonly string[],
    private readonly _names: Readonly<Record<number, string>>,
    private readonly _inputSize: readonly [number, number],
    private readonly _confThreshold: number,
    private readonly _iouThreshold: number,
    private readonly _maxDetections: number,
  ) {
    super(session);
  }

  private _pipelineCache: LetterboxPipeline | null = null;

  /**
   * Run the model once on a zero-filled tensor, paying one-time costs up front.
   *
   * The first inference of a session is not representative: WebGPU compiles its
   * shaders on it and the WASM backend faults in its arenas, which on a phone
   * can turn the first frame into seconds while every later frame is tens of
   * milliseconds. Calling this while a loading spinner is still up moves that
   * cost somewhere the user is already waiting.
   *
   * @param runs How many warm-up inferences to run. One is enough for WASM;
   *   WebGPU sometimes settles on the second.
   */
  async warmup(runs: number = 1): Promise<void> {
    const [tw, th] = this._inputSize;
    for (let i = 0; i < runs; i++) {
      const tensor = toFloat32Tensor(zeroTensorData(tw, th), [1, 3, th, tw]);
      await this._session.run({ [this._session.inputName]: tensor });
    }
  }

  /**
   * The fused preprocessing pipeline, built on first use.
   *
   * Lazily, because constructing it allocates canvases: a task built in an
   * environment without a canvas implementation stays constructible, and only
   * fails if it is actually asked to preprocess something.
   */
  private get _pipeline(): LetterboxPipeline {
    if (this._pipelineCache === null) {
      this._pipelineCache = new LetterboxPipeline(this._inputSize[0], this._inputSize[1]);
    }
    return this._pipelineCache;
  }

  /** Load the model and resolve labels. */
  static async create(
    model: ModelSource,
    options: DetectorOptions = {},
  ): Promise<Detector> {
    const head: DetectorHead = options.head ?? "yolo";
    if (head !== "yolo") {
      throw new Error(`Unsupported detector head '${head}'. Supported: 'yolo'.`);
    }
    const session = await OrtSession.create(model, options);
    const labels = resolveLabels(options.labels ?? modelNames(session.metadata) ?? "coco", {
      numClasses: options.numClasses ?? detectionNumClasses(session.outputShape) ?? undefined,
    });
    const names: Record<number, string> = {};
    for (let i = 0; i < labels.length; i++) {
      names[i] = labels[i] as string;
    }
    return new Detector(
      session,
      head,
      labels,
      names,
      resolveInputSize({
        graphShape: session.inputShape,
        requested: options.inputSize,
        fallback: [640, 640],
      }),
      options.confThreshold ?? 0.25,
      options.iouThreshold ?? 0.45,
      options.maxDetections ?? 300,
    );
  }

  /** The decoder family used to interpret the model's output. */
  get head(): DetectorHead {
    return this._head;
  }

  /** Class labels indexed by class id. */
  get labels(): readonly string[] {
    return this._labels;
  }

  /** Class id → class name dict (matches Ultralytics' `model.names`). */
  get names(): Readonly<Record<number, string>> {
    return this._names;
  }

  /**
   * The `[width, height]` this task preprocesses to.
   *
   * Resolved at creation time from the model's graph when it declares a static
   * input, so reading it back tells you the resolution inference really runs at
   * — not merely what was requested.
   */
  get inputSize(): readonly [number, number] {
    return this._inputSize;
  }

  /** Number of classes the model predicts. */
  get numClasses(): number {
    return this._labels.length;
  }

  /**
   * Alias for {@link predict} — call the detector like a torch `nn.Module`.
   *
   * Use as `det.call(img)` since JavaScript class instances are not callable;
   * for direct invocation, prefer `det.predict(img)`. The full
   * {@link DetectorPredictOptions} (including `classes`) is supported.
   */
  async call(
    image: ImageInput,
    options: DetectorPredictOptions = {},
  ): Promise<DetectionResults[]> {
    return this.predict(image, options);
  }

  /**
   * Run detection on a single image.
   *
   * The returned envelope carries a {@link Speed} breakdown in `speed`,
   * mirroring Ultralytics' `results[0].speed`.
   */
  async predict(
    image: ImageInput,
    options: DetectorPredictOptions = {},
  ): Promise<DetectionResults[]> {
    const timer = new SpeedTimer();
    const path = typeof image === "string" ? image : null;
    const original = await loadImage(image);
    timer.stage("load");
    const { tensor, scale, padLeft, padTop } = this._preprocess(original);
    timer.stage("preprocess");
    const outputs = await this._session.run({ [this._session.inputName]: tensor });
    this._pipeline.release();
    timer.stage("inference");

    const firstOutputName = this._session.outputNames[0];
    if (firstOutputName === undefined) {
      throw new Error("Detector model has no outputs.");
    }
    const raw = outputs[firstOutputName];
    if (raw === undefined) {
      throw new Error(`Detector model output ${firstOutputName} missing from run() result.`);
    }

    const decodedAll = decodeYolo(raw.data as Float32Array, raw.dims, {
      originalWidth: original.width,
      originalHeight: original.height,
      padLeft,
      padTop,
      scale,
      confThreshold: options.confThreshold ?? this._confThreshold,
      iouThreshold: options.iouThreshold ?? this._iouThreshold,
      maxDetections: this._maxDetections,
    });

    const decoded =
      options.classes !== undefined
        ? (() => {
            const allowed = new Set(options.classes);
            return decodedAll.filter((d) => allowed.has(d.classId));
          })()
        : decodedAll;

    const detections = decoded.map((d) =>
      this._buildResult(original, d.bbox, d.classId, d.confidence),
    );

    const orig: readonly [number, number] = [original.height, original.width];
    const boxes = this._buildBoxes(detections, orig);
    timer.stage("postprocess");
    return [
      new DetectionResults(
        boxes,
        detections,
        this._names,
        original,
        orig,
        path,
        timer.speed(),
      ),
    ];
  }

  /**
   * Letterbox and pack the image into the tensor the model expects.
   *
   * Runs through {@link LetterboxPipeline}, which fuses the resize, the
   * padding and the HWC-to-CHW float conversion into one `drawImage` plus one
   * readback loop, and reuses its output buffer between frames. The buffer is
   * handed straight to ONNX Runtime, so {@link _pipeline.release} must not be
   * called until the run resolves.
   */
  private _preprocess(image: RGBImage): {
    tensor: ort.Tensor;
    scale: number;
    padLeft: number;
    padTop: number;
  } {
    const [tw, th] = this._inputSize;
    const fused = this._pipeline.run(image);
    return {
      tensor: toFloat32Tensor(fused.data, [1, 3, th, tw]),
      scale: fused.scale,
      padLeft: fused.padLeft,
      padTop: fused.padTop,
    };
  }

  private _buildResult(
    original: RGBImage,
    bbox: BoundingBox,
    classId: number,
    confidence: number,
  ): DetectionResult {
    const [x1, y1, x2, y2] = bbox.asIntXyxy();
    const cx1 = Math.max(0, x1);
    const cy1 = Math.max(0, y1);
    const cx2 = Math.min(original.width, x2);
    const cy2 = Math.min(original.height, y2);

    let cropped: RGBImage;
    if (cx2 > cx1 && cy2 > cy1) {
      const cw = cx2 - cx1;
      const ch = cy2 - cy1;
      const out = new Uint8Array(cw * ch * 3);
      for (let row = 0; row < ch; row++) {
        const srcOffset = ((cy1 + row) * original.width + cx1) * 3;
        out.set(
          original.data.subarray(srcOffset, srcOffset + cw * 3),
          row * cw * 3,
        );
      }
      cropped = new RGBImage(out, cw, ch);
    } else {
      cropped = new RGBImage(new Uint8Array(0), 0, 0);
    }

    const className = this._names[classId] ?? `class_${classId}`;

    return {
      classId,
      className,
      confidence,
      bbox,
      cls: classId,
      name: className,
      conf: confidence,
      box: bbox,
      croppedImage: cropped,
    };
  }

  private _buildBoxes(
    detections: readonly DetectionResult[],
    origShape: readonly [number, number],
  ): Boxes {
    const n = detections.length;
    const xyxy = new Float32Array(n * 4);
    const cls = new Int32Array(n);
    const conf = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const d = detections[i] as DetectionResult;
      xyxy[i * 4] = d.bbox.x1;
      xyxy[i * 4 + 1] = d.bbox.y1;
      xyxy[i * 4 + 2] = d.bbox.x2;
      xyxy[i * 4 + 3] = d.bbox.y2;
      cls[i] = d.classId;
      conf[i] = d.confidence;
    }
    return new Boxes(xyxy, cls, conf, origShape);
  }
}
