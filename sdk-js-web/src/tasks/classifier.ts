/**
 * Image classification task using ONNX Runtime Web.
 */

import type * as ort from "onnxruntime-web";

import { type ModelSource, type OrtSessionOptions, OrtSession } from "../core/session.js";
import { type ImageInput, loadImage } from "../io/image.js";
import { type LabelSpec, resolveLabels } from "../labels.js";
import { softmax, topK } from "../postprocess/classification.js";
import {
  normalize,
  resize,
  toCHW,
  toFloat32Tensor,
} from "../preprocess/image.js";
import { ClassificationResults, Probs } from "../results.js";
import { VisionTask } from "./base.js";
import {
  type ClassProbability,
  type ClassificationResult,
  type RGBImage,
} from "../types.js";

const IMAGENET_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
const IMAGENET_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

export interface ClassifierOptions extends OrtSessionOptions {
  /** Class label spec — see {@link resolveLabels}. */
  readonly labels: LabelSpec;
  /**
   * Number of classes the model can predict. Required when `labels` is `null`
   * or when you want to validate that the supplied labels match the model.
   */
  readonly numClasses?: number;
  /** Model input `[width, height]` in pixels. Defaults to `[224, 224]`. */
  readonly inputSize?: readonly [number, number];
  /** Per-channel RGB mean used for normalization. Defaults to ImageNet. */
  readonly mean?: readonly [number, number, number];
  /** Per-channel RGB standard deviation. Defaults to ImageNet. */
  readonly std?: readonly [number, number, number];
  /**
   * If `true` (default), apply softmax to the raw model output. Set to
   * `false` for models whose final layer already produces a probability
   * distribution.
   */
  readonly applySoftmax?: boolean;
}

export interface ClassifierPredictOptions {
  /**
   * If set, the per-class probability list in `results[0].result.probabilities`
   * is truncated to the top-K entries. The bulk `probs` view always exposes
   * the full vector.
   */
  readonly topK?: number;
}

/**
 * Image classifier wrapping an ONNX model with ImageNet-style preprocessing.
 *
 * `predict()` returns `Promise<ClassificationResults[]>` (length 1 for a
 * single image), mirroring Ultralytics' API. The envelope exposes a `probs`
 * collection (`top1`, `top1conf`, `top5`, `top5conf`, `data`) plus the
 * legacy per-class probability list with names already resolved.
 *
 * Defaults: 224×224 RGB input, `float32` normalized with ImageNet mean/std,
 * NCHW layout, batch size 1, softmax applied to the raw output.
 *
 * @example
 * ```typescript
 * const clf = await Classifier.create("/models/resnet50.onnx", {
 *   labels: ["tench", "goldfish", ...]  // 1000 ImageNet labels
 * });
 * const r = (await clf.predict("/images/dog.jpg"))[0];
 * console.log(r.cls, r.conf, r.name);
 * console.log(r.probs.top5, r.probs.top5conf);
 * ```
 */
export class Classifier extends VisionTask {
  private constructor(
    session: OrtSession,
    private readonly _labels: readonly string[],
    private readonly _names: Readonly<Record<number, string>>,
    private readonly _inputSize: readonly [number, number],
    private readonly _mean: readonly [number, number, number],
    private readonly _std: readonly [number, number, number],
    private readonly _applySoftmax: boolean,
  ) {
    super(session);
  }

  /** Load the model and resolve labels. */
  static async create(
    model: ModelSource,
    options: ClassifierOptions,
  ): Promise<Classifier> {
    const session = await OrtSession.create(model, options);
    const labels = resolveLabels(options.labels, { numClasses: options.numClasses });
    const names: Record<number, string> = {};
    for (let i = 0; i < labels.length; i++) {
      names[i] = labels[i] as string;
    }
    return new Classifier(
      session,
      labels,
      names,
      options.inputSize ?? [224, 224],
      options.mean ?? IMAGENET_MEAN,
      options.std ?? IMAGENET_STD,
      options.applySoftmax ?? true,
    );
  }

  /** Class labels indexed by class id. */
  get labels(): readonly string[] {
    return this._labels;
  }

  /** Class id → class name dict (matches Ultralytics' `model.names`). */
  get names(): Readonly<Record<number, string>> {
    return this._names;
  }

  /** Number of classes the model can predict. */
  get numClasses(): number {
    return this._labels.length;
  }

  /** Alias for {@link predict} (parity with PyTorch `nn.Module.__call__`). */
  async call(
    image: ImageInput,
    options: ClassifierPredictOptions = {},
  ): Promise<ClassificationResults[]> {
    return this.predict(image, options);
  }

  /** Run classification on a single image. */
  async predict(
    image: ImageInput,
    options: ClassifierPredictOptions = {},
  ): Promise<ClassificationResults[]> {
    const path = typeof image === "string" ? image : null;
    const original = await loadImage(image);
    const tensor = this._preprocess(original);
    const outputs = await this._session.run({ [this._session.inputName]: tensor });
    const firstOutputName = this._session.outputNames[0];
    if (firstOutputName === undefined) {
      throw new Error("Classifier model has no outputs.");
    }
    const raw = outputs[firstOutputName];
    if (raw === undefined) {
      throw new Error(`Classifier model output ${firstOutputName} missing from run() result.`);
    }
    const fullProbs = this._postprocess(raw.data as Float32Array);

    const { indices, values } = topK(fullProbs, options.topK ?? null);
    const probabilities: ClassProbability[] = [];
    for (let i = 0; i < indices.length; i++) {
      const id = indices[i] as number;
      const className = this._labels[id] ?? `class_${id}`;
      probabilities.push({
        classId: id,
        className,
        probability: values[i] as number,
        cls: id,
        name: className,
        conf: values[i] as number,
      });
    }
    if (probabilities.length === 0) {
      throw new Error("Classifier produced no probabilities (empty output).");
    }

    const top = probabilities[0] as ClassProbability;
    const result: ClassificationResult = {
      classId: top.classId,
      className: top.className,
      confidence: top.probability,
      cls: top.classId,
      name: top.className,
      conf: top.probability,
      image: original,
      probabilities,
    };

    const orig: readonly [number, number] = [original.height, original.width];
    return [
      new ClassificationResults(
        new Probs(fullProbs),
        result,
        this._names,
        original,
        orig,
        path,
      ),
    ];
  }

  private _preprocess(image: RGBImage): ort.Tensor {
    const [tw, th] = this._inputSize;
    const resized = resize(image, tw, th);
    const normalized = normalize(resized, this._mean, this._std);
    const chw = toCHW(normalized, resized.width, resized.height, 3);
    return toFloat32Tensor(chw, [1, 3, resized.height, resized.width]);
  }

  private _postprocess(raw: Float32Array): Float32Array {
    return this._applySoftmax ? softmax(raw) : new Float32Array(raw);
  }
}
