/**
 * Thin wrapper around `onnxruntime-web` `InferenceSession` with typed metadata.
 */

import type * as ort from "onnxruntime-web";
import * as ortRuntime from "onnxruntime-web";

import { InferenceError, ModelLoadError } from "./exceptions.js";
import { type DeclaredShape, declaredShapesFrom } from "./graph.js";
import { readModelMetadata } from "./metadata.js";
import { resolveProviders } from "./providers.js";

/** Anything `InferenceSession.create` accepts. */
export type ModelSource = string | ArrayBufferLike | Uint8Array;

/**
 * Fetch a model URL as bytes so its metadata can be read.
 *
 * Falls back to the URL itself when the fetch fails, letting ORT try its own
 * load path: losing the metadata map is a downgrade, but failing to load a model
 * that ORT could have fetched would be a regression.
 *
 * @param url Where the `.onnx` lives.
 * @returns The model bytes, or the original URL when they could not be fetched.
 */
async function fetchModel(url: string): Promise<Uint8Array | string> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      warnMetadataUnavailable(url, `HTTP ${response.status} ${response.statusText}`);
      return url;
    }
    return new Uint8Array(await response.arrayBuffer());
  } catch (err) {
    warnMetadataUnavailable(url, (err as Error).message);
    return url;
  }
}

/**
 * Warn that a model's metadata could not be read, and say what that costs.
 *
 * The fallback itself is right — losing the metadata beats failing a load that
 * ORT could have completed on its own — but it used to be silent, and the
 * symptom it produces is remote from the cause: class names come back as
 * `class_0`, `class_1`, ... with nothing anywhere explaining why. Whoever hits
 * this needs to be told that passing `labels` is the way out.
 *
 * @param url The model URL that could not be fetched here.
 * @param reason What went wrong, as reported by `fetch`.
 */
function warnMetadataUnavailable(url: string, reason: string): void {
  console.warn(
    `[@ort-vision-sdk/web] Could not fetch ${url} to read its metadata (${reason}). ` +
      "Letting ONNX Runtime load it instead: the model will work, but its baked-in " +
      "class names are unavailable, so labels fall back to class_0, class_1, ... " +
      "Pass `labels` explicitly to name them.",
  );
}

export interface OrtSessionOptions {
  /** Execution providers in preference order. `undefined` uses {@link DEFAULT_PROVIDERS}. */
  readonly providers?: readonly string[];
  /** Optional ORT session options forwarded to `InferenceSession.create`. */
  readonly sessionOptions?: ort.InferenceSession.SessionOptions;
  /**
   * Whether to read the model's custom metadata map (`names`, `task`, `imgsz`).
   * Defaults to `true`.
   *
   * The runtime does not expose that map, so it is read from the file itself —
   * which means a URL model is fetched here and handed to ORT as bytes instead
   * of letting ORT fetch it. That is the same single download either way, and
   * it is what lets a task resolve its labels off the model. Set to `false` to
   * keep the URL path untouched and leave {@link OrtSession.metadata} empty.
   *
   * `false` is also the escape hatch when a device cannot afford the bytes: the
   * fetched buffer is dropped before ORT builds the graph (see
   * {@link OrtSession.create}), but ORT's own load path still keeps the model out
   * of reach of anything the SDK holds. A session built this way resolves its
   * input size from the graph as usual — only the class names are lost, so a
   * caller taking this route has to pass `labels` itself.
   */
  readonly readMetadata?: boolean;
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
    private readonly _metadata: Readonly<Record<string, string>>,
  ) {}

  /**
   * Load an ONNX model into an ORT inference session.
   *
   * The metadata map is read **before** the session is built, and that order is
   * load-bearing on memory-constrained devices. ORT copies the model into its
   * WASM heap and then allocates the graph and the weights on top of that copy;
   * a `readModelMetadata` call placed after `InferenceSession.create` keeps the
   * JavaScript-side buffer reachable across the whole build, so a 5 MB model
   * costs 5 MB of JS heap plus 5 MB of WASM heap plus the weights at the same
   * instant. Reading first makes the buffer collectable as soon as ORT has copied
   * it — on a phone that was the difference between a session and
   * `Can't create a session. failed to allocate a buffer of size N`.
   *
   * @param model Either a URL string, or a `Uint8Array`/`ArrayBuffer` containing the model bytes.
   * @param options Provider list, pass-through `SessionOptions`, and whether to
   *   read the model's metadata map (see {@link OrtSessionOptions.readMetadata}).
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
    const wantsMetadata = options.readMetadata !== false;
    const source =
      typeof model === "string" && wantsMetadata ? await fetchModel(model) : model;
    const metadata =
      wantsMetadata && typeof source !== "string" ? readModelMetadata(source) : {};

    let session: ort.InferenceSession;
    try {
      if (typeof source === "string") {
        session = await ortRuntime.InferenceSession.create(source, sessionOptions);
      } else if (source instanceof Uint8Array) {
        session = await ortRuntime.InferenceSession.create(source, sessionOptions);
      } else {
        session = await ortRuntime.InferenceSession.create(
          source as ArrayBuffer,
          sessionOptions,
        );
      }
    } catch (err) {
      throw new ModelLoadError(
        `Failed to load ONNX model: ${(err as Error).message}`,
        { cause: err },
      );
    }

    return new OrtSession(session, providers, metadata);
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
   * Shapes the graph declares for its outputs, in declaration order.
   *
   * Dynamic (symbolic) axes appear as `null`. Reading them is how a task can
   * tell how many classes a head emits without being told.
   */
  get outputShapes(): readonly DeclaredShape[] {
    return declaredShapesFrom(
      this._session.outputMetadata as
        | readonly ort.InferenceSession.ValueMetadata[]
        | undefined,
    );
  }

  /**
   * Shape the graph declares for its first output, dynamic axes as `null`.
   *
   * Empty when the runtime reports no metadata for it.
   */
  get outputShape(): DeclaredShape {
    return this.outputShapes[0] ?? [];
  }

  /**
   * The model's custom metadata map — `names`, `task`, `imgsz`, ... for an
   * Ultralytics export.
   *
   * Read from the model's bytes at load time, since the runtime does not expose
   * it. Empty when the session was created with `readMetadata: false`, from a
   * URL that could not be fetched here, or from a model carrying no metadata.
   */
  get metadata(): Readonly<Record<string, string>> {
    return this._metadata;
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
