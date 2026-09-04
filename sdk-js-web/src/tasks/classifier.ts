/**
 * Image classification task using ONNX Runtime Web.
 */

import type * as ort from "onnxruntime-web";

import { type ModelSource, type OrtSessionOptions, OrtSession } from "../core/session.js";
import { SpeedTimer } from "../core/timing.js";
import { type ImageInput, loadImage } from "../io/image.js";
import { classificationNumClasses, resolveInputSize } from "../core/graph.js";
import { modelNames } from "../core/metadata.js";
import { type LabelSpec, resolveLabels } from "../labels.js";
import {
  type Normalization,
  resolveNormalization,
} from "../normalization.js";
import { softmax, topK } from "../postprocess/classification.js";
import { toFloat32Tensor } from "../preprocess/image.js";
import { ResizePipeline, zeroTensorData } from "../preprocess/pipeline.js";
import { ClassificationResults, Probs } from "../results.js";
import { VisionTask } from "./base.js";
import {
  type ClassProbability,
  type ClassificationResult,
  type RGBImage,
} from "../types.js";

export interface ClassifierOptions extends OrtSessionOptions {
  /**
   * Class label spec — see {@link resolveLabels}.
   *
   * Optional: when omitted, the names the export baked into the model are used
   * (Ultralytics writes them as `names` in the metadata map). Only when the
   * model carries none does this fall back to generated `class_<id>` labels.
   * Passing a spec always wins, for a model whose names are wrong or absent.
   */
  readonly labels?: LabelSpec;
  /**
   * Number of classes the model can predict.
   *
   * Optional: inferred from the classification head's declared output shape
   * `(B, nc)`. Pass it to validate that the supplied labels match the model.
   */
  readonly numClasses?: number;
  /**
   * Model input `[width, height]` in pixels.
   *
   * Only used when the model's graph leaves its spatial axes dynamic: a graph
   * that declares a static size always wins, since that is the only shape ONNX
   * Runtime will accept. Defaults to `[224, 224]`.
   */
  readonly inputSize?: readonly [number, number];
  /**
   * Which preprocessing this model expects — see {@link Normalization}.
   *
   * Defaults to `"auto"`, which reads the model's own export metadata and picks
   * `"ultralytics"` (raw `[0, 1]`) for an Ultralytics classification head and
   * `"imagenet"` for everything else.
   */
  readonly normalization?: Normalization;
  /** Per-channel RGB mean, overriding the preset. Defaults to the preset's. */
  readonly mean?: readonly [number, number, number];
  /** Per-channel RGB standard deviation, overriding the preset. */
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
    private readonly _normalization: string,
  ) {
    super(session);
  }

  /**
   * Which preprocessing this classifier applies to every image.
   *
   * One of the {@link Normalization} preset names, or `"custom"` when the caller
   * supplied `mean`/`std` directly. Worth reading when a model underperforms:
   * feeding a classifier a differently prepared tensor than it was trained on
   * degrades it without throwing anything, so "what does this assume" is the
   * first question.
   */
  get normalization(): string {
    return this._normalization;
  }

  private _pipelineCache: ResizePipeline | null = null;

  /**
   * Run the model once on a zero tensor, paying one-time costs up front.
   *
   * The first inference of a session is not representative: WebGPU compiles its
   * shaders on it and the WASM backend faults in its arenas. Calling this while
   * a loading spinner is still up moves that cost somewhere the user is already
   * waiting — which matters most for a classifier running as the last step of
   * an on-device analysis, where the delay lands right before the answer shows.
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
   * Lazily, because constructing it reserves the output buffer: a task built in
   * an environment without a canvas implementation stays constructible, and only
   * fails if it is actually asked to preprocess something.
   */
  private get _pipeline(): ResizePipeline {
    if (this._pipelineCache === null) {
      const [tw, th] = this._inputSize;
      this._pipelineCache = new ResizePipeline(tw, th, this._mean, this._std);
    }
    return this._pipelineCache;
  }

  /**
   * Load the model, resolve labels, and settle the preprocessing.
   *
   * @param model The model source — a URL, or the bytes.
   * @param options Labels, input size, normalization, and session options.
   * @throws {RangeError} If `normalization` names an unknown preset, or names
   *   one while `mean`/`std` are also given.
   */
  static async create(
    model: ModelSource,
    options: ClassifierOptions = {},
  ): Promise<Classifier> {
    const session = await OrtSession.create(model, options);
    const normalization = resolveNormalization(session.metadata, {
      normalization: options.normalization,
      mean: options.mean,
      std: options.std,
    });
    const numClasses =
      options.numClasses ?? classificationNumClasses(session.outputShape) ?? undefined;
    const labels = resolveLabels(options.labels ?? modelNames(session.metadata), {
      numClasses,
    });
    const names: Record<number, string> = {};
    for (let i = 0; i < labels.length; i++) {
      names[i] = labels[i] as string;
    }
    return new Classifier(
      session,
      labels,
      names,
      resolveInputSize({
        graphShape: session.inputShape,
        requested: options.inputSize,
        fallback: [224, 224],
      }),
      normalization.mean,
      normalization.std,
      options.applySoftmax ?? true,
      normalization.name,
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
    const timer = new SpeedTimer();
    const path = typeof image === "string" ? image : null;
    const original = await loadImage(image);
    timer.stage("load");
    const tensor = this._preprocess(original);
    timer.stage("preprocess");
    const outputs = await this._session.run({ [this._session.inputName]: tensor });
    this._pipeline.release();
    timer.stage("inference");
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
    const probs = new Probs(fullProbs);
    timer.stage("postprocess");
    return [
      new ClassificationResults(
        probs,
        result,
        this._names,
        original,
        orig,
        path,
        timer.speed(),
      ),
    ];
  }

  private _preprocess(image: RGBImage): ort.Tensor {
    const [tw, th] = this._inputSize;
    const { data } = this._pipeline.run(image);
    return toFloat32Tensor(data, [1, 3, th, tw]);
  }

  private _postprocess(raw: Float32Array): Float32Array {
    return this._applySoftmax ? softmax(raw) : new Float32Array(raw);
  }
}
