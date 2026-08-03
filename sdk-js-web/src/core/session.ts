/**
 * Thin wrapper around `onnxruntime-web` `InferenceSession` with typed metadata.
 */

import type * as ort from "onnxruntime-web";
import * as ortRuntime from "onnxruntime-web";

import { InferenceError, ModelLoadError } from "./exceptions.js";
import { type DeclaredShape, declaredShapesFrom } from "./graph.js";
import { resolveProviders } from "./providers.js";

/** Anything `InferenceSession.create` accepts. */
export type ModelSource = string | ArrayBufferLike | Uint8Array;

export interface OrtSessionOptions {
  /** Execution providers in preference order. `undefined` uses {@link DEFAULT_PROVIDERS}. */
  readonly providers?: readonly string[];
  /** Optional ORT session options forwarded to `InferenceSession.create`. */
  readonly sessionOptions?: ort.InferenceSession.SessionOptions;
}

/**
 * Wrap an ONNX Runtime Web `InferenceSession` with convenient metadata access.
 *
 * The wrapper exposes input/output names and the shapes the graph declares,
 * manages execution-provider selection, provides a typed {@link OrtSession.run}
 * method, and releases the native session through {@link OrtSession.release}.
 */
export class OrtSession {
  private constructor(
    private readonly _session: ort.InferenceSession,
    public readonly providers: readonly string[],
  ) {}

  /**
   * Load an ONNX model into an ORT inference session.
   *
   * @param model Either a URL string fetched by ORT, or a `Uint8Array`/`ArrayBuffer` containing the model bytes.
   * @param options Provider list and pass-through `SessionOptions`.
   * @throws {@link ModelLoadError} if the model cannot be loaded.
   */
  static async create(
    model: ModelSource,
    options: OrtSessionOptions = {},
  ): Promise<OrtSession> {
    const providers = resolveProviders(options.providers);
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      ...(options.sessionOptions ?? {}),
      executionProviders: providers as ort.InferenceSession.SessionOptions["executionProviders"],
    };

    let session: ort.InferenceSession;
    try {
      if (typeof model === "string") {
        session = await ortRuntime.InferenceSession.create(model, sessionOptions);
      } else if (model instanceof Uint8Array) {
        session = await ortRuntime.InferenceSession.create(model, sessionOptions);
      } else {
        session = await ortRuntime.InferenceSession.create(
          model as ArrayBuffer,
          sessionOptions,
        );
      }
    } catch (err) {
      throw new ModelLoadError(
        `Failed to load ONNX model: ${(err as Error).message}`,
        { cause: err },
      );
    }

    return new OrtSession(session, providers);
  }

  /** Names of the model's inputs, in declaration order. */
  get inputNames(): readonly string[] {
    return this._session.inputNames;
  }

  /** Name of the first (and usually only) input. */
  get inputName(): string {
    const name = this._session.inputNames[0];
    if (name === undefined) {
      throw new InferenceError("Model has no inputs.");
    }
    return name;
  }

  /** Names of the model's outputs, in declaration order. */
  get outputNames(): readonly string[] {
    return this._session.outputNames;
  }

  /**
   * Shapes the graph declares for its inputs, in declaration order.
   *
   * Dynamic (symbolic) axes appear as `null`. Empty shapes mean the runtime
   * reported no metadata — either a non-tensor input, or an `onnxruntime-web`
   * older than 1.21, which predates input metadata.
   */
  get inputShapes(): readonly DeclaredShape[] {
    return declaredShapesFrom(
      this._session.inputMetadata as
        | readonly ort.InferenceSession.ValueMetadata[]
        | undefined,
    );
  }

  /**
   * Shape the graph declares for its first input, dynamic axes as `null`.
   *
   * Empty when the runtime reports no metadata for it.
   */
  get inputShape(): DeclaredShape {
    return this.inputShapes[0] ?? [];
  }

  /**
   * Release the native session and free its memory.
   *
   * Call it when a session is discarded while the page lives on — rebuilding a
   * task at a different input size, swapping in a newer model. A failure from
   * the runtime is ignored: a session being torn down has nothing left to fail
   * at, and the caller is already moving on.
   */
  async release(): Promise<void> {
    await this._session.release().catch(() => undefined);
  }

  /** The underlying `onnxruntime-web` session, for advanced use cases. */
  get raw(): ort.InferenceSession {
    return this._session;
  }

  /**
   * Run inference and return all outputs.
   *
   * @param feeds Map of input name to `ort.Tensor`. Keys must match {@link inputNames}.
   * @throws {@link InferenceError} if ORT raises any error during execution.
   */
  async run(
    feeds: Record<string, ort.Tensor>,
  ): Promise<Record<string, ort.Tensor>> {
    try {
      const result = await this._session.run(feeds);
      return result as Record<string, ort.Tensor>;
    } catch (err) {
      throw new InferenceError(
        `Inference failed: ${(err as Error).message}`,
        { cause: err },
      );
    }
  }
}
