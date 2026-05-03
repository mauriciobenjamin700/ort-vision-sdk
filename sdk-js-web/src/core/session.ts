/**
 * Thin wrapper around `onnxruntime-web` `InferenceSession` with typed metadata.
 */

import type * as ort from "onnxruntime-web";
import * as ortRuntime from "onnxruntime-web";

import { InferenceError, ModelLoadError } from "./exceptions.js";
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
 * The wrapper exposes input/output names, manages execution-provider
 * selection, and provides a typed {@link OrtSession.run} method.
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
