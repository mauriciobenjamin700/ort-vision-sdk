/**
 * Instance-segmentation task using YOLO seg ONNX models (v8-seg / v11-seg / ...).
 */

import type * as ort from "onnxruntime-web";

import {
  type ModelSource,
  type OrtSessionOptions,
  OrtSession,
} from "../core/session.js";
import { SpeedTimer } from "../core/timing.js";
import { resolveInputSize } from "../core/graph.js";
import { type ImageInput, loadImage } from "../io/image.js";
import { type LabelSpec, resolveLabels } from "../labels.js";
import { decodeYoloSeg } from "../postprocess/segmentation.js";
import {
  letterbox,
  toCHW,
  toFloat32,
  toFloat32Tensor,
} from "../preprocess/image.js";
import { Boxes, Masks, SegmentationResults } from "../results.js";
import { VisionTask } from "./base.js";
import {
  type BoundingBox,
  type SegmentationResult,
  Mask,
  RGBImage,
} from "../types.js";

/**
 * Decoder family for the segmentation head.
 *
 * - `"yolo-seg"`: YOLO instance-segmentation head with two outputs —
 *   `[1, 4 + nc + nm, N]` per-anchor predictions plus `[1, nm, mh, mw]`
 *   prototype masks. Covers YOLOv8-seg, v11-seg, v26-seg.
 *
 * The SDK does **not** auto-detect this — the caller is responsible for
 * picking a head that matches their export.
 */
export type SegmenterHead = "yolo-seg";

export interface SegmenterOptions extends OrtSessionOptions {
  /**
   * Decoder family for the segmentation head. Default `"yolo-seg"` covers
   * YOLOv8-seg/v11-seg/v26-seg.
   */
  readonly head?: SegmenterHead;
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
  /** Maximum number of instances per image. */
  readonly maxDetections?: number;
  /** Probability cutoff applied to soft masks. Defaults to `0.5`. */
  readonly maskThreshold?: number;
}

export interface SegmenterPredictOptions {
  readonly confThreshold?: number;
  readonly iouThreshold?: number;
  /**
   * If set, keep only instances whose `classId` is in this list.
   * Mirrors Ultralytics' `model.predict(img, classes=[0, 16])`.
   */
  readonly classes?: readonly number[];
}

/**
 * Instance segmenter for YOLO seg ONNX models (v8-seg / v11-seg / ...).
 *
 * The model is expected to expose two outputs:
 *
 * 1. `output0`: `(1, 4 + numClasses + numMaskCoefs, numAnchors)` — per-anchor
 *    predictions (boxes, class scores, mask coefficients).
 * 2. `output1`: `(1, numMaskCoefs, maskH, maskW)` — prototype masks.
 *
 * `predict()` returns `Promise<SegmentationResults[]>` (length 1 for a
 * single image), mirroring Ultralytics' API. The envelope exposes:
 *
 * - `boxes`: bulk numpy view (`xyxy`, `xywh`, `xyxyn`, `xywhn`, `cls`, `conf`).
 * - `masks`: per-instance binary masks cropped to each box.
 * - per-instance {@link SegmentationResult} via iteration.
 *
 * @example
 * ```typescript
 * const seg = await Segmenter.create("/models/yolov8n-seg.onnx");
 * const r = (await seg.predict("/images/street.jpg"))[0];
 * for (const inst of r) {
 *   console.log(inst.cls, inst.conf, inst.box.xyxy);
 * }
 * ```
 */
export class Segmenter extends VisionTask {
  private constructor(
    session: OrtSession,
    private readonly _head: SegmenterHead,
    private readonly _labels: readonly string[],
    private readonly _names: Readonly<Record<number, string>>,
    private readonly _inputSize: readonly [number, number],
    private readonly _confThreshold: number,
    private readonly _iouThreshold: number,
    private readonly _maxDetections: number,
    private readonly _maskThreshold: number,
  ) {
    super(session);
  }

  /** Load the model and resolve labels. */
  static async create(
    model: ModelSource,
    options: SegmenterOptions = {},
  ): Promise<Segmenter> {
    const head: SegmenterHead = options.head ?? "yolo-seg";
    if (head !== "yolo-seg") {
      throw new Error(`Unsupported segmenter head '${head}'. Supported: 'yolo-seg'.`);
    }
    const session = await OrtSession.create(model, options);
    const labels = resolveLabels(options.labels ?? "coco", {
      numClasses: options.numClasses,
    });
    const names: Record<number, string> = {};
    for (let i = 0; i < labels.length; i++) {
      names[i] = labels[i] as string;
    }
    return new Segmenter(
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
      options.maskThreshold ?? 0.5,
    );
  }

  /** The decoder family used to interpret the model's output. */
  get head(): SegmenterHead {
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

  /** Alias for {@link predict} (parity with PyTorch `nn.Module.__call__`). */
  async call(
    image: ImageInput,
    options: SegmenterPredictOptions = {},
  ): Promise<SegmentationResults[]> {
    return this.predict(image, options);
  }

  /** Run instance segmentation on a single image. */
  async predict(
    image: ImageInput,
    options: SegmenterPredictOptions = {},
  ): Promise<SegmentationResults[]> {
    const timer = new SpeedTimer();
    const path = typeof image === "string" ? image : null;
    const original = await loadImage(image);
    timer.stage("load");
    const { tensor, scale, padLeft, padTop } = this._preprocess(original);
    timer.stage("preprocess");
    const outputs = await this._session.run({ [this._session.inputName]: tensor });
    timer.stage("inference");

    const { perAnchor, prototypes } = this._splitOutputs(outputs);

    const decodedAll = decodeYoloSeg(
      perAnchor.data as Float32Array,
      perAnchor.dims,
      prototypes.data as Float32Array,
      prototypes.dims,
      {
        numClasses: this._labels.length,
        inputWidth: this._inputSize[0],
        inputHeight: this._inputSize[1],
        originalWidth: original.width,
        originalHeight: original.height,
        padLeft,
        padTop,
        scale,
        confThreshold: options.confThreshold ?? this._confThreshold,
        iouThreshold: options.iouThreshold ?? this._iouThreshold,
        maxDetections: this._maxDetections,
        maskThreshold: this._maskThreshold,
      },
    );

    const decoded =
      options.classes !== undefined
        ? (() => {
            const allowed = new Set(options.classes);
            return decodedAll.filter((d) => allowed.has(d.classId));
          })()
        : decodedAll;

    const detections = decoded.map((d) =>
      this._buildResult(original, d.bbox, d.classId, d.confidence, d.mask),
    );

    const orig: readonly [number, number] = [original.height, original.width];
    const boxes = this._buildBoxes(detections, orig);
    const masks = this._buildMasks(detections, orig);
    timer.stage("postprocess");
    return [
      new SegmentationResults(
        boxes,
        masks,
        detections,
        this._names,
        original,
        orig,
        path,
        timer.speed(),
      ),
    ];
  }

  private _preprocess(image: RGBImage): {
    tensor: ort.Tensor;
    scale: number;
    padLeft: number;
    padTop: number;
  } {
    const [tw, th] = this._inputSize;
    const lb = letterbox(image, tw, th);
    const f32 = toFloat32(lb.image);
    const chw = toCHW(f32, lb.image.width, lb.image.height, 3);
    return {
      tensor: toFloat32Tensor(chw, [1, 3, lb.image.height, lb.image.width]),
      scale: lb.scale,
      padLeft: lb.padLeft,
      padTop: lb.padTop,
    };
  }

  private _splitOutputs(
    outputs: Record<string, ort.Tensor>,
  ): { perAnchor: ort.Tensor; prototypes: ort.Tensor } {
    let perAnchor: ort.Tensor | undefined;
    let prototypes: ort.Tensor | undefined;
    for (const name of this._session.outputNames) {
      const t = outputs[name];
      if (t === undefined) continue;
      if (t.dims.length === 3 && perAnchor === undefined) {
        perAnchor = t;
      } else if (t.dims.length === 4 && prototypes === undefined) {
        prototypes = t;
      }
    }
    if (perAnchor === undefined || prototypes === undefined) {
      const shapes = this._session.outputNames.map(
        (n) => `${n}: ${JSON.stringify(outputs[n]?.dims ?? [])}`,
      );
      throw new Error(
        `Segmenter expected one 3-D and one 4-D output, got [${shapes.join(", ")}].`,
      );
    }
    return { perAnchor, prototypes };
  }

  private _buildResult(
    original: RGBImage,
    bbox: BoundingBox,
    classId: number,
    confidence: number,
    mask: Mask,
  ): SegmentationResult {
    const [x1, y1, x2, y2] = bbox.asIntXyxy();
    const cx1 = Math.max(0, x1);
    const cy1 = Math.max(0, y1);
    const cx2 = Math.min(original.width, x2);
    const cy2 = Math.min(original.height, y2);

    let segmentedImage: RGBImage;
    let finalMask = mask;
    if (cx2 > cx1 && cy2 > cy1 && mask.data.length > 0) {
      const cropW = cx2 - cx1;
      const cropH = cy2 - cy1;
      const mw = Math.min(mask.width, cropW);
      const mh = Math.min(mask.height, cropH);
      const segData = new Uint8Array(mw * mh * 3);
      for (let row = 0; row < mh; row++) {
        const srcRowOffset = ((cy1 + row) * original.width + cx1) * 3;
        const dstRowOffset = row * mw * 3;
        const maskRowOffset = row * mask.width;
        for (let col = 0; col < mw; col++) {
          const m = mask.data[maskRowOffset + col];
          if (m !== 0) {
            const s = srcRowOffset + col * 3;
            const d = dstRowOffset + col * 3;
            segData[d] = original.data[s];
            segData[d + 1] = original.data[s + 1];
            segData[d + 2] = original.data[s + 2];
          }
        }
      }
      segmentedImage = new RGBImage(segData, mw, mh);
      if (mw !== mask.width || mh !== mask.height) {
        const trimmed = new Uint8Array(mw * mh);
        for (let row = 0; row < mh; row++) {
          trimmed.set(
            mask.data.subarray(row * mask.width, row * mask.width + mw),
            row * mw,
          );
        }
        finalMask = new Mask(trimmed, mw, mh);
      }
    } else {
      finalMask = new Mask(new Uint8Array(0), 0, 0);
      segmentedImage = new RGBImage(new Uint8Array(0), 0, 0);
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
      mask: finalMask,
      segmentedImage,
    };
  }

  private _buildBoxes(
    detections: readonly SegmentationResult[],
    origShape: readonly [number, number],
  ): Boxes {
    const n = detections.length;
    const xyxy = new Float32Array(n * 4);
    const cls = new Int32Array(n);
    const conf = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const d = detections[i] as SegmentationResult;
      xyxy[i * 4] = d.bbox.x1;
      xyxy[i * 4 + 1] = d.bbox.y1;
      xyxy[i * 4 + 2] = d.bbox.x2;
      xyxy[i * 4 + 3] = d.bbox.y2;
      cls[i] = d.classId;
      conf[i] = d.confidence;
    }
    return new Boxes(xyxy, cls, conf, origShape);
  }

  private _buildMasks(
    detections: readonly SegmentationResult[],
    origShape: readonly [number, number],
  ): Masks {
    const xyxy = new Float32Array(detections.length * 4);
    for (let i = 0; i < detections.length; i++) {
      const d = detections[i] as SegmentationResult;
      xyxy[i * 4] = d.bbox.x1;
      xyxy[i * 4 + 1] = d.bbox.y1;
      xyxy[i * 4 + 2] = d.bbox.x2;
      xyxy[i * 4 + 3] = d.bbox.y2;
    }
    return new Masks(
      detections.map((d) => d.mask),
      xyxy,
      origShape,
    );
  }
}
