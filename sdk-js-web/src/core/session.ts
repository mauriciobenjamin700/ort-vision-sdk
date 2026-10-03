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
import { readModelInputTypes, readModelMetadata, readModelShapes } from "./metadata.js";
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

/**
 * An 84-byte ONNX model — one `Identity` on a float `[1]` — used to start the runtime.
 *
 * Generated with `onnx.helper` (opset 13, IR 7, producer `ort-vision-sdk`).
 * See {@link warmRuntime} for why it exists.
 */
const RUNTIME_PROBE_MODEL =
  "CAcSDm9ydC12aXNpb24tc2RrOjoKEAoBeBIBeSIISWRlbnRpdHkSBHdhcm1aDwoBeBIKCggIARIECgIIAWIPCgF5EgoKCAgBEgQKAggBQgQKABAN";

/** Runtime start-ups already begun on this page, keyed by provider names. */
const runtimeWarmups = new Map<string, Promise<void>>();

/**
 * Start ONNX Runtime's backends for a provider list, once per page.
 *
 * ORT downloads and compiles its WebAssembly binary — 12.8 MB for the plain
 * WASM build, 25.9 MB for the one carrying WebGPU — inside the first
 * `InferenceSession.create`. {@link OrtSession.create} only reaches that call
 * after it has downloaded the model, so on a real network the two largest
 * downloads of a page load ran one after the other. Creating and releasing a
 * session on a tiny model while the real one downloads lets them overlap; the
 * real `create` then finds the runtime ready.
 *
 * A failure here is swallowed: the real `create` runs against the same runtime
 * and reports the same problem with the right model in the message.
 *
 * @param specs Providers the real session will use.
 * @returns Settles when the runtime is up, or when starting it failed.
 */
function warmRuntime(specs: readonly ProviderSpec[]): Promise<void> {
  const key = specs.map(providerName).join(",");
  let pending = runtimeWarmups.get(key);
  if (pending === undefined) {
    pending = (async () => {
      try {
        const probe = Uint8Array.from(atob(RUNTIME_PROBE_MODEL), (c) => c.charCodeAt(0));
        const session = await ortRuntime.InferenceSession.create(probe, {
          executionProviders: specs as ort.InferenceSession.SessionOptions["executionProviders"],
        });
        await session.release();
      } catch {
        return;
      }
    })();
    runtimeWarmups.set(key, pending);
  }
  return pending;
}

/**
 * Warn that a model optimized offline is about to run on WebGPU.
 *
 * `optimize_model` fuses operators for the CPU, which is what the WASM backend
 * runs. On WebGPU some of those fused nodes have no GPU kernel: on a
 * YOLO11n-seg, ONNX Runtime placed 2 nodes on the CPU for the optimized file
 * and none for the original, so every inference pays GPU↔CPU copies. The SDK
 * cannot swap the file, so it says which one to ship.
 */
function warnPreOptimizedOnWebGpu(): void {
  console.warn(
    "[@ort-vision-sdk/web] This model was pre-optimized by optimize_model, which targets the " +
      "WASM backend; on WebGPU some of its fused nodes fall back to the CPU. For WebGPU, ship " +
      "the original export instead.",
  );
}

/**
 * Metadata key the Python SDK's `quantize_model` writes on an INT8 model.
 *
 * Mirrored by `QUANTIZATION_KEY` in the Python SDK; both test suites pin the
 * string. See {@link keepQuantizedOffWebGpu} for what a session does with it.
 */
export const QUANTIZATION_KEY = "ort_vision_sdk.quantization";

/**
 * Drop `webgpu` from the providers of a model `quantize_model` produced.
 *
 * ONNX Runtime Web's WebGPU `DequantizeLinear` rejects the quantized bias such
 * a model carries — `scale and zero-point inputs must have the same rank` —
 * so the first `predict()` fails; and a variant with the bias left in float
 * ran but returned different detections than WASM for the same file. The
 * model is fine on WASM, which is also where INT8 pays (52.7 ms vs 68.4 ms on
 * a YOLO11n-seg), so the session runs there and says so.
 *
 * @param specs Providers that survived detection.
 * @param metadata The model's metadata map.
 * @returns `specs` without `webgpu` when the model is marked quantized; the
 *   WASM fallback when nothing else is left.
 */
function keepQuantizedOffWebGpu(
  specs: readonly ProviderSpec[],
  metadata: Readonly<Record<string, string>>,
): ProviderSpec[] {
  if (metadata[QUANTIZATION_KEY] === undefined) return [...specs];
  const kept = specs.filter((spec) => providerName(spec) !== "webgpu");
  if (kept.length === specs.length) return kept;
  console.warn(
    `[@ort-vision-sdk/web] This model was quantized by quantize_model (${metadata[QUANTIZATION_KEY]}); ` +
      "ONNX Runtime Web's WebGPU backend cannot run its quantized operators, so it runs on WASM.",
  );
  return kept.length > 0 ? kept : [FALLBACK_PROVIDER];
}

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
    /**
     * Shapes the model file declares, keyed by value name.
     *
     * The fallback for {@link inputShapes} and {@link outputShapes} when the
     * runtime reports none, which `onnxruntime-web` does below 1.22. Empty when
     * the bytes were never in hand (a URL loaded with `readMetadata: false`).
     */
    private readonly _fileShapes: {
      readonly inputs: Readonly<Record<string, DeclaredShape>>;
      readonly outputs: Readonly<Record<string, DeclaredShape>>;
    } = { inputs: {}, outputs: {} },
  ) {}

  /**
   * Tail of the run queue: settles when the latest queued run has finished.
   *
   * See {@link run} for why runs are serialized.
   */
  private _runQueue: Promise<unknown> = Promise.resolve();

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
   * When the model is downloaded here, ORT's runtime is started alongside the
   * download too — see {@link warmRuntime}.
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
    const detection = detectProviders(requested).then((detected) =>
      detected.length > 0 ? detected : [FALLBACK_PROVIDER],
    );
    const [detected, source] = await Promise.all([
      detection,
      shouldFetch ? fetchModel(model, cacheName) : Promise.resolve(model),
      shouldFetch ? detection.then(warmRuntime) : Promise.resolve(),
    ]);
    const metadata =
      wantsMetadata && typeof source !== "string" ? readModelMetadata(source) : {};
    const specs = keepQuantizedOffWebGpu(detected, metadata);
    const requestedNames = requested.map(providerName);
    const providers = specs.map(providerName);
    if (options.providers !== undefined && options.providers.length > 0) {
      warnOnDroppedProviders(requestedNames, detected.map(providerName));
    }
    if (metadata[GRAPH_OPTIMIZATION_KEY] !== undefined && providers.includes("webgpu")) {
      warnPreOptimizedOnWebGpu();
    }
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
    const fileShapes =
      typeof source !== "string" ? readModelShapes(source) : { inputs: {}, outputs: {} };
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

    return new OrtSession(session, providers, metadata, requestedNames, inputTypes, fileShapes);
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
   * Dynamic (symbolic) axes appear as `null`. Read from the runtime when it
   * reports them (`onnxruntime-web` >= 1.22), otherwise from the model file. An
   * empty shape means neither source declared one: a non-tensor input, or a URL
   * model loaded with `readMetadata: false` on a runtime that reports nothing.
   */
  get inputShapes(): readonly DeclaredShape[] {
    return this._shapesOf(
      this._session.inputMetadata as
        | readonly ort.InferenceSession.ValueMetadata[]
        | undefined,
      this._session.inputNames,
      this._fileShapes.inputs,
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
   * tell how many classes a head emits without being told. Same sources as
   * {@link inputShapes}.
   */
  get outputShapes(): readonly DeclaredShape[] {
    return this._shapesOf(
      this._session.outputMetadata as
        | readonly ort.InferenceSession.ValueMetadata[]
        | undefined,
      this._session.outputNames,
      this._fileShapes.outputs,
    );
  }

  /**
   * Pick the runtime's declared shapes, or the file's when the runtime has none.
   *
   * @param metadata What the runtime reports, `undefined` below 1.22.
   * @param names The session's value names, in its own order.
   * @param fromFile Shapes the model file declares, keyed by name.
   * @returns One shape per name, in the session's order.
   */
  private _shapesOf(
    metadata: readonly ort.InferenceSession.ValueMetadata[] | undefined,
    names: readonly string[],
    fromFile: Readonly<Record<string, DeclaredShape>>,
  ): readonly DeclaredShape[] {
    const reported = declaredShapesFrom(metadata);
    if (reported.length > 0) return reported;
    return names.map((name) => fromFile[name] ?? []);
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
   * Runs on one session are serialized. ONNX Runtime Web refuses a second
   * `run` while one is in flight — it rejects with `Session already started` —
   * so two overlapping `predict()` calls on the same task used to fail. Queued
   * here, they overlap where they can: while one run executes (in a worker,
   * with `env.wasm.proxy`, or on the GPU), the next `predict()` decodes and
   * preprocesses its frame and only waits for its turn at the runtime. A run
   * that fails does not block the ones behind it.
   *
   * @param feeds Map of input name to `ort.Tensor`. Keys must match {@link inputNames}.
   * @throws {@link InferenceError} if ORT raises any error during execution.
   */
  run(feeds: Record<string, ort.Tensor>): Promise<Record<string, ort.Tensor>> {
    const turn = this._runQueue.then(() => this._runNow(feeds));
    this._runQueue = turn.catch(() => undefined);
    return turn;
  }

  /**
   * Execute one run immediately; only {@link run}'s queue calls this.
   *
   * @param feeds Map of input name to `ort.Tensor`.
   * @throws {@link InferenceError} if ORT raises any error during execution.
   */
  private async _runNow(
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
