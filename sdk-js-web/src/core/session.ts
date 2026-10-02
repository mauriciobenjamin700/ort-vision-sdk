/**
 * Thin wrapper around `onnxruntime-web` `InferenceSession` with typed metadata.
 */

import type * as ort from "onnxruntime-web";
import * as ortRuntime from "onnxruntime-web";

import {
  DEFAULT_TENSOR_TYPE,
  hasFloat16Array,
  tensorTypeFor,
  toFeedData,
} from "./dtypes.js";
import { InferenceError, ModelLoadError } from "./exceptions.js";
import { type DeclaredShape, declaredShapesFrom } from "./graph.js";
import { readModelInputTypes, readModelMetadata } from "./metadata.js";
import {
  FALLBACK_PROVIDER,
  type ProviderSpec,
  detectProviders,
  providerName,
  resolveProviders,
} from "./providers.js";

/** Anything `InferenceSession.create` accepts. */
export type ModelSource = string | ArrayBufferLike | Uint8Array;

/**
 * Metadata key naming the level a model was graph-optimized at offline.
 *
 * Written by the Python SDK's `optimize_model`; both test suites pin the same
 * string, so a rename on one side fails. See {@link OrtSession.create} for what
 * the session does when it finds it.
 */
export const GRAPH_OPTIMIZATION_KEY = "ort_vision_sdk.graph_optimization";

/** Cache Storage bucket used when {@link OrtSessionOptions.cache} is `true`. */
export const DEFAULT_MODEL_CACHE = "ort-vision-sdk-models";

/**
 * Fetch a model URL as bytes so its metadata can be read.
 *
 * Falls back to the URL itself when the fetch fails, letting ORT try its own
 * load path: losing the metadata map is a downgrade, but failing to load a model
 * that ORT could have fetched would be a regression.
 *
 * @param url Where the `.onnx` lives.
 * @param cacheName Cache Storage bucket to read from and fill, or `null` to
 *   always go to the network.
 * @returns The model bytes, or the original URL when they could not be fetched.
 */
async function fetchModel(url: string, cacheName: string | null): Promise<Uint8Array | string> {
  const cache = cacheName === null ? null : await openCache(cacheName);
  if (cache !== null) {
    const hit = await cache.match(url).catch(() => undefined);
    if (hit !== undefined) return new Uint8Array(await hit.arrayBuffer());
  }
  try {
    const response = await fetch(url);
    if (!response.ok) {
      warnMetadataUnavailable(url, `HTTP ${response.status} ${response.statusText}`);
      return url;
    }
    const buffer = await response.arrayBuffer();
    if (cache !== null) await storeModel(cache, url, buffer);
    return new Uint8Array(buffer);
  } catch (err) {
    warnMetadataUnavailable(url, (err as Error).message);
    return url;
  }
}

/**
 * Put a model's bytes into the cache, never failing the load over it.
 *
 * Stores a fresh `Response` built from bytes already read rather than a
 * `clone()` of the network response: one read, and nothing that can throw
 * synchronously outside the guard. Awaited on purpose — a `put` left running
 * would keep its copy of the model reachable while ORT builds the session, the
 * same doubled peak {@link OrtSession.create} orders its reads to avoid.
 *
 * @param cache The open bucket.
 * @param url Cache key.
 * @param buffer The model.
 */
async function storeModel(cache: Cache, url: string, buffer: ArrayBuffer): Promise<void> {
  try {
    await cache.put(url, new Response(buffer));
  } catch {
    return;
  }
}

/**
 * Open a Cache Storage bucket, or report that there is none to open.
 *
 * Cache Storage only exists in secure contexts (HTTPS, `localhost`), and a
 * browser may refuse it outright — private windows, blocked site data, an
 * exhausted quota. None of that is worth failing a model load over: the model
 * is fetched from the network as if caching had not been asked for. A failed
 * `put` is swallowed for the same reason — see {@link storeModel}.
 *
 * @param name Bucket name.
 * @returns The cache, or `null` when this environment cannot provide one.
 */
async function openCache(name: string): Promise<Cache | null> {
  if (typeof caches === "undefined") return null;
  try {
    return await caches.open(name);
  } catch {
    return null;
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
  /**
   * Execution providers in preference order. `undefined` uses {@link DEFAULT_PROVIDERS}.
   *
   * Each entry is a name (`"webgpu"`, `"wasm"`) or ORT's config object for that
   * provider, which passes provider options through untouched —
   * `{ name: "webgpu", preferredLayout: "NHWC" }`.
   *
   * Naming one explicitly also opts into a `console.warn` when this browser
   * cannot offer it, instead of falling back in silence.
   */
  readonly providers?: readonly ProviderSpec[];
  /**
   * Keep a URL model in the browser's Cache Storage between page loads.
   *
   * `true` uses the {@link DEFAULT_MODEL_CACHE} bucket; a string names the
   * bucket. A later `create` with the same URL reads the bytes from the cache
   * instead of the network — for a model of several megabytes, the bulk of
   * what a returning visitor waits for. Defaults to `false`.
   *
   * The URL is the cache key and nothing expires it: publish a changed model
   * under a new URL (`yolo.v2.onnx`, `?v=2`) or a new bucket name, or delete
   * the bucket with `caches.delete(name)`. Ignored for models passed as bytes,
   * and silently skipped where Cache Storage is unavailable (non-secure
   * contexts, blocked site data).
   */
  readonly cache?: boolean | string;
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
   * keep the URL path untouched (unless {@link cache} is set, which needs the
   * bytes) and leave {@link OrtSession.metadata} empty.
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
 * Warn when a provider named explicitly by the caller is not going to run.
 *
 * Only explicit requests are worth a warning. The default list exists precisely
 * so that falling from `webgpu` to `wasm` is the expected outcome — but a caller
 * who wrote `providers: ["webgpu"]` and lands on WASM has a page several times
 * slower than intended and nothing in the console to explain it.
 *
 * @param requested Providers that were asked for.
 * @param effective Providers that survived capability detection.
 */
function warnOnDroppedProviders(
  requested: readonly string[],
  effective: readonly string[],
): void {
  const dropped = requested.filter((provider) => !effective.includes(provider));
  if (dropped.length === 0) {
    return;
  }
  console.warn(
    `This browser cannot offer the requested execution provider(s) ${JSON.stringify(dropped)}; ` +
      `the session will run on ${JSON.stringify(effective)}. Inference still produces correct ` +
      "results, on the fallback provider.",
  );
}

/**
 * Name the tensor type of every graph input, straight from the model bytes.
 *
 * @param model The `.onnx` file contents.
 * @returns Input name → tensor type name. Empty when the file carries no
 *   readable graph, which callers read as "assume float32".
 */
function declaredInputTypes(
  model: Uint8Array | ArrayBufferLike,
): Readonly<Record<string, string>> {
  const declared = readModelInputTypes(model);
  const named: Record<string, string> = {};
  for (const [name, elemType] of Object.entries(declared)) {
    named[name] = tensorTypeFor(elemType);
  }
  return named;
}

/**
 * Warn when the model's bytes were in hand but no input type came back.
 *
 * Every ONNX graph declares at least one input, so an empty result from bytes
 * means the reader failed, not that the model said nothing. Without this the
 * failure is invisible: the session falls back to float32, which is right for
 * most models and wrong for exactly the half-precision ones this reader exists
 * to support — so it looks like nothing is broken until a `predict()` throws
 * `Unexpected input data type`. That is how a 1 MB ceiling on the graph
 * descent shipped in 0.9.0 and reached a consumer.
 */
function warnOnUnreadableTypes(): void {
  console.warn(
    "Could not read any input type from this model's bytes; feeds will be built as float32. " +
      "A half-precision model will fail at inference with 'Unexpected input data type'. " +
      "Please report this with the model that produced it.",
  );
}

/**
 * Refuse a half-precision model in an environment that cannot feed one.
 *
 * ORT requires a real `Float16Array` for a `float16` tensor — the same bits in
 * a `Uint16Array` are rejected — and not every browser has one yet. Without
 * this check the session builds happily and the first `predict()` throws from
 * inside the preprocessing, which is both later and further from the cause.
 *
 * @param inputTypes Tensor type per input name.
 * @throws {@link ModelLoadError} when the graph wants half precision and the
 *   runtime has no `Float16Array`.
 */
function assertFeedableTypes(inputTypes: Readonly<Record<string, string>>): void {
  const half = Object.entries(inputTypes).filter(([, type]) => type === "float16");
  if (half.length === 0 || hasFloat16Array()) return;
  const names = half.map(([name]) => name).join(", ");
  throw new ModelLoadError(
    `This model declares half-precision input(s) [${names}], but this environment has no ` +
      "Float16Array, which ONNX Runtime requires for a float16 tensor. Use a float32 export " +
      "of the model, or run in a browser that supports Float16Array.",
  );
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
    /**
     * Execution providers this session is expected to run on.
     *
     * The requested list narrowed to what this browser can actually offer — a
     * `webgpu` entry survives only where an adapter exists. Best-effort: ORT-Web
     * exposes no way to ask which provider a session ended up on, so an entry
     * here means "not ruled out", not "confirmed". See
     * {@link requestedProviders} for what was asked for.
     *
     * When nothing survives — a caller asking for `webgpu` alone on a device
     * without it — this falls back to {@link FALLBACK_PROVIDER}, which ORT-Web
     * can always run. Handing ORT the unsatisfiable list instead makes
     * `InferenceSession.create` reject with "no available backend found", so the
     * page gets no inference at all rather than the slow-but-working fallback
     * the `console.warn` describes. Measured in a real Chromium, where
     * `navigator.gpu` exists but yields no adapter.
     */
    public readonly providers: readonly string[],
    private readonly _metadata: Readonly<Record<string, string>>,
    /**
     * Execution providers that were asked for, after defaults were applied.
     *
     * Kept separate because ORT-Web falls back silently: a page that asks for
     * `webgpu` on a device without it runs on WASM and is told nothing.
     */
    public readonly requestedProviders: readonly string[],
    /**
     * Tensor type each input declares, keyed by input name.
     *
     * Read from the model file, not from the session: `inputMetadata` is
     * `undefined` on `onnxruntime-web` 1.20.1. Empty when the bytes were never
     * in hand — a URL loaded with `readMetadata: false` — in which case every
     * feed is built as float32, the behaviour before this existed.
     */
    private readonly _inputTypes: Readonly<Record<string, string>> = {},
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
   * The provider probe and the model download run concurrently: neither
   * depends on the other, and on a real GPU `requestAdapter()` is not free.
   *
   * A model whose metadata carries {@link GRAPH_OPTIMIZATION_KEY} was already
   * optimized offline by the Python SDK's `optimize_model`, so it is loaded
   * with `graphOptimizationLevel: "disabled"` instead of paying the optimizer a
   * second time — measured on a YOLO11n-seg under WASM, 29 ms of session
   * creation down to 12 ms with unchanged inference time. An explicit
   * `graphOptimizationLevel` in `sessionOptions` always wins. The mark is read
   * with the rest of the metadata, so `readMetadata: false` skips it too.
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
    const requested = resolveProviders(options.providers);
    const wantsMetadata = options.readMetadata !== false;
    const cacheName =
      options.cache === true ? DEFAULT_MODEL_CACHE : options.cache ? options.cache : null;
    const shouldFetch = typeof model === "string" && (wantsMetadata || cacheName !== null);
    const [detected, source] = await Promise.all([
      detectProviders(requested),
      shouldFetch ? fetchModel(model, cacheName) : Promise.resolve(model),
    ]);
    const specs = detected.length > 0 ? detected : [FALLBACK_PROVIDER];
    const requestedNames = requested.map(providerName);
    const providers = specs.map(providerName);
    if (options.providers !== undefined && options.providers.length > 0) {
      warnOnDroppedProviders(requestedNames, providers);
    }
    const metadata =
      wantsMetadata && typeof source !== "string" ? readModelMetadata(source) : {};
    const preOptimized =
      metadata[GRAPH_OPTIMIZATION_KEY] !== undefined &&
      options.sessionOptions?.graphOptimizationLevel === undefined;
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      ...(preOptimized ? { graphOptimizationLevel: "disabled" as const } : {}),
      ...(options.sessionOptions ?? {}),
      executionProviders: specs as ort.InferenceSession.SessionOptions["executionProviders"],
    };
    const inputTypes =
      typeof source !== "string" ? declaredInputTypes(source) : {};
    if (typeof source !== "string" && Object.keys(inputTypes).length === 0) {
      warnOnUnreadableTypes();
    }
    assertFeedableTypes(inputTypes);

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

    return new OrtSession(session, providers, metadata, requestedNames, inputTypes);
  }

  /**
   * Tensor type each input declares, in declaration order.
   *
   * `"float32"` for a normal export, `"float16"` for one exported with
   * `half=True`. ORT matches a feed's type against this exactly, so the tasks
   * read it and convert at the feed boundary.
   */
  get inputDtypes(): readonly string[] {
    return this.inputNames.map((name) => this._inputTypes[name] ?? DEFAULT_TENSOR_TYPE);
  }

  /** Tensor type the first input declares. */
  get inputDtype(): string {
    return this._inputTypes[this.inputName] ?? DEFAULT_TENSOR_TYPE;
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
      const result = await this._session.run(this._asDeclaredTypes(feeds));
      return result as Record<string, ort.Tensor>;
    } catch (err) {
      throw new InferenceError(
        `Inference failed: ${(err as Error).message}`,
        { cause: err },
      );
    }
  }

  /**
   * Rebuild any feed whose type does not match what its input declares.
   *
   * ORT matches feed types against the graph exactly: a float32 tensor against
   * a `half=True` export fails the run with `Unexpected input data type`. Every
   * preprocessing path in this SDK produces `Float32Array`, on purpose — the
   * normalization arithmetic belongs in single precision — so the conversion
   * happens here, once, at the only boundary all of them pass through. A feed
   * that already carries the declared type is handed on untouched, so a caller
   * building its own half tensor pays nothing.
   *
   * @param feeds The tensors to run with.
   * @returns The same mapping, with mismatched float32 feeds converted.
   */
  private _asDeclaredTypes(
    feeds: Record<string, ort.Tensor>,
  ): Record<string, ort.Tensor> {
    let converted: Record<string, ort.Tensor> | null = null;
    for (const [name, tensor] of Object.entries(feeds)) {
      const declared = this._inputTypes[name];
      if (declared === undefined || declared === tensor.type) continue;
      if (!(tensor.data instanceof Float32Array)) continue;
      const data = toFeedData(tensor.data, declared);
      if (data === tensor.data) continue;
      converted ??= { ...feeds };
      converted[name] = new ortRuntime.Tensor(
        declared as "float16",
        data as unknown as Float32Array,
        tensor.dims as number[],
      );
    }
    return converted ?? feeds;
  }
}
