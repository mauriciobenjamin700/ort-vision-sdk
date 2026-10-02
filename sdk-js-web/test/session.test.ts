import { afterEach, describe, expect, it, vi } from "vitest";

import type * as MetadataModule from "../src/core/metadata.js";

/**
 * Where the model buffer is read relative to the session build.
 *
 * `OrtSession.create` fetches a URL model into a `Uint8Array` so the metadata map
 * can be read out of it, and ORT then copies that buffer into its WASM heap
 * before allocating the graph and the weights on top. Holding the JavaScript
 * buffer across that build doubles the cost of every model at its peak, which on
 * a phone surfaced as `Can't create a session. failed to allocate a buffer of
 * size N`. So the read has to happen *before* `InferenceSession.create`, and the
 * order is pinned here rather than left to whoever edits the function next.
 */

/** Order in which the metadata read and the ORT session build were reached. */
const calls: string[] = [];

/** What each `InferenceSession.create` call was handed as its model. */
const sources: unknown[] = [];

/** Byte length of the runtime probe model `OrtSession.create` warms ORT with. */
const PROBE_LENGTH = 84;

const createSession = vi.fn((model: unknown, _options?: unknown) => {
  if (model instanceof Uint8Array && model.length === PROBE_LENGTH) {
    calls.push("probe");
    return Promise.resolve({ release: () => Promise.resolve() });
  }
  calls.push("create");
  sources.push(model);
  return Promise.resolve({
    inputNames: ["images"],
    outputNames: ["output0"],
    inputMetadata: undefined,
    outputMetadata: undefined,
    release: () => Promise.resolve(),
  });
});

vi.mock("onnxruntime-web", () => ({
  InferenceSession: {
    get create() {
      return createSession;
    },
  },
}));

vi.mock("../src/core/metadata.js", async (importOriginal) => {
  const actual = await importOriginal<typeof MetadataModule>();
  return {
    ...actual,
    readModelMetadata: (model: Uint8Array | ArrayBufferLike) => {
      calls.push("metadata");
      return actual.readModelMetadata(model);
    },
  };
});

const { OrtSession } = await import("../src/core/session.js");

/** Field number of `metadata_props` in `ModelProto`. */
const METADATA_PROPS_FIELD = 14;

/**
 * Encode a base-128 varint.
 *
 * @param value Non-negative integer to encode.
 * @returns Its varint bytes.
 */
function varint(value: number): number[] {
  const out: number[] = [];
  let remaining = value;
  while (remaining > 0x7f) {
    out.push((remaining & 0x7f) | 0x80);
    remaining = Math.floor(remaining / 128);
  }
  out.push(remaining);
  return out;
}

/**
 * Encode a length-delimited protobuf field.
 *
 * @param field Field number.
 * @param payload Bytes of the field's value.
 * @returns The encoded field.
 */
function lengthDelimited(field: number, payload: readonly number[]): number[] {
  return [...varint((field << 3) | 2), ...varint(payload.length), ...payload];
}

/**
 * Build a minimal `ModelProto` carrying only a metadata map.
 *
 * @param entries Metadata key/value pairs.
 * @returns The encoded model bytes.
 */
function modelProto(entries: Readonly<Record<string, string>>): Uint8Array {
  const encoder = new TextEncoder();
  const bytes: number[] = [];
  for (const [key, value] of Object.entries(entries)) {
    const entry = [
      ...lengthDelimited(1, [...encoder.encode(key)]),
      ...lengthDelimited(2, [...encoder.encode(value)]),
    ];
    bytes.push(...lengthDelimited(METADATA_PROPS_FIELD, entry));
  }
  return new Uint8Array(bytes);
}

/**
 * Serve one model file over a stubbed `fetch`.
 *
 * @param model The bytes the URL should answer with.
 */
function serveModel(model: Uint8Array): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve({
        ok: true,
        arrayBuffer: () => Promise.resolve(model.buffer),
      }),
    ),
  );
}

afterEach(() => {
  calls.length = 0;
  sources.length = 0;
  createSession.mockClear();
  vi.unstubAllGlobals();
});

describe("OrtSession.create", () => {
  it("reads the model metadata before ORT builds the session", async () => {
    serveModel(modelProto({ names: "{0: 'ocular-mucosa'}" }));

    const session = await OrtSession.create("/models/detect.onnx");

    expect(calls.filter((call) => call !== "probe")).toEqual(["metadata", "create"]);
    expect(session.metadata.names).toBe("{0: 'ocular-mucosa'}");
  });

  it("hands ORT the fetched bytes, not the URL", async () => {
    serveModel(modelProto({ task: "detect" }));

    await OrtSession.create("/models/detect.onnx");

    expect(sources[0]).toBeInstanceOf(Uint8Array);
  });

  it("skips the fetch entirely when metadata is not wanted", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);

    await OrtSession.create("/models/detect.onnx", { readMetadata: false });

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(calls).toEqual(["create"]);
    expect(sources[0]).toBe("/models/detect.onnx");
  });
});

/**
 * A stand-in for the Cache Storage API, backed by a `Map`.
 *
 * @param seed URL → bytes already in the bucket.
 * @returns The fake `caches` global, plus the bucket names opened and the URLs put.
 */
function fakeCaches(seed: Readonly<Record<string, Uint8Array>> = {}) {
  const store = new Map<string, Uint8Array>(Object.entries(seed));
  const opened: string[] = [];
  const put: string[] = [];
  const cache = {
    match: (url: string) => {
      const hit = store.get(url);
      return Promise.resolve(
        hit === undefined ? undefined : { arrayBuffer: () => Promise.resolve(hit.buffer) },
      );
    },
    put: (url: string) => {
      put.push(url);
      return Promise.resolve();
    },
  };
  return {
    global: { open: (name: string) => (opened.push(name), Promise.resolve(cache)) },
    opened,
    put,
  };
}

describe("OrtSession.create with cache", () => {
  it("reads a cached model without touching the network", async () => {
    const caches = fakeCaches({ "/models/detect.onnx": modelProto({ task: "detect" }) });
    vi.stubGlobal("caches", caches.global);
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);

    const session = await OrtSession.create("/models/detect.onnx", { cache: true });

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(caches.opened).toEqual(["ort-vision-sdk-models"]);
    expect(session.metadata.task).toBe("detect");
  });

  it("fetches and stores a model the bucket does not hold yet", async () => {
    const caches = fakeCaches();
    vi.stubGlobal("caches", caches.global);
    const model = modelProto({ task: "detect" });
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve({
          ok: true,
          arrayBuffer: () => Promise.resolve(model.buffer),
        }),
      ),
    );

    await OrtSession.create("/models/detect.onnx", { cache: "my-models" });

    expect(caches.opened).toEqual(["my-models"]);
    expect(caches.put).toEqual(["/models/detect.onnx"]);
  });

  it("still fetches the bytes when metadata is off but caching is on", async () => {
    const caches = fakeCaches();
    vi.stubGlobal("caches", caches.global);
    serveModel(modelProto({}));

    await OrtSession.create("/models/detect.onnx", { cache: true, readMetadata: false });

    expect(sources[0]).toBeInstanceOf(Uint8Array);
  });

  it("falls back to the network where Cache Storage does not exist", async () => {
    serveModel(modelProto({ task: "detect" }));

    const session = await OrtSession.create("/models/detect.onnx", { cache: true });

    expect(session.metadata.task).toBe("detect");
  });
});

describe("OrtSession.create with a pre-optimized model", () => {
  /** The session options ORT was handed on the latest `create`. */
  function lastSessionOptions(): Record<string, unknown> {
    const call = createSession.mock.calls.at(-1) as unknown[] | undefined;
    return (call?.[1] ?? {}) as Record<string, unknown>;
  }

  it("pins the metadata key the Python optimize_model writes", async () => {
    const { GRAPH_OPTIMIZATION_KEY } = await import("../src/core/session.js");
    expect(GRAPH_OPTIMIZATION_KEY).toBe("ort_vision_sdk.graph_optimization");
  });

  it("skips ORT's optimizer when the model says it was optimized offline", async () => {
    serveModel(modelProto({ "ort_vision_sdk.graph_optimization": "extended" }));

    await OrtSession.create("/models/detect.onnx");

    expect(lastSessionOptions().graphOptimizationLevel).toBe("disabled");
  });

  it("lets an explicit optimization level win over the mark", async () => {
    serveModel(modelProto({ "ort_vision_sdk.graph_optimization": "extended" }));

    await OrtSession.create("/models/detect.onnx", {
      sessionOptions: { graphOptimizationLevel: "all" },
    });

    expect(lastSessionOptions().graphOptimizationLevel).toBe("all");
  });

  it("warns when the pre-optimized model is headed for WebGPU", async () => {
    vi.stubGlobal("navigator", { gpu: { requestAdapter: () => Promise.resolve({}) } });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    serveModel(modelProto({ "ort_vision_sdk.graph_optimization": "extended" }));

    await OrtSession.create("/models/detect.onnx", { providers: ["webgpu", "wasm"] });

    expect(warn.mock.calls.some((c) => String(c[0]).includes("ship the original"))).toBe(true);
    warn.mockRestore();
  });

  it("leaves an unmarked model on ORT's default level", async () => {
    serveModel(modelProto({ task: "detect" }));

    await OrtSession.create("/models/detect.onnx");

    expect(lastSessionOptions().graphOptimizationLevel).toBeUndefined();
  });
});

describe("OrtSession.create with provider config objects", () => {
  it("hands ORT the config objects and reports provider names", async () => {
    serveModel(modelProto({}));

    const session = await OrtSession.create("/models/detect.onnx", {
      providers: [{ name: "wasm", custom: 1 }],
    });

    const call = createSession.mock.calls.at(-1) as unknown[];
    expect((call[1] as Record<string, unknown>).executionProviders).toEqual([
      { name: "wasm", custom: 1 },
    ]);
    expect(session.providers).toEqual(["wasm"]);
    expect(session.requestedProviders).toEqual(["wasm"]);
  });
});

describe("OrtSession.run", () => {
  /**
   * Build a session whose runtime refuses overlapping runs, like ORT-Web does.
   *
   * @param fail Run indices (0-based) that should reject.
   * @returns The session and the order runs started and finished in.
   */
  async function strictSession(fail: readonly number[] = []) {
    const events: string[] = [];
    let active = false;
    let count = 0;
    createSession.mockImplementationOnce((model: unknown) => {
      sources.push(model);
      return Promise.resolve({
        inputNames: ["images"],
        outputNames: ["output0"],
        inputMetadata: undefined,
        outputMetadata: undefined,
        release: () => Promise.resolve(),
        run: async () => {
          if (active) throw new Error("Session already started");
          const index = count++;
          active = true;
          events.push(`start ${index}`);
          await new Promise((resolve) => setTimeout(resolve, 5));
          events.push(`end ${index}`);
          active = false;
          if (fail.includes(index)) throw new Error(`run ${index} failed`);
          return { output0: index };
        },
      });
    });
    const session = await OrtSession.create(new Uint8Array(0), { readMetadata: false });
    return { session, events };
  }

  it("serializes overlapping runs instead of letting ORT refuse them", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { session, events } = await strictSession();
    warn.mockRestore();

    const results = await Promise.all([session.run({}), session.run({}), session.run({})]);

    expect(results.map((r) => r.output0)).toEqual([0, 1, 2]);
    expect(events).toEqual(["start 0", "end 0", "start 1", "end 1", "start 2", "end 2"]);
  });

  it("keeps running the queue after a run fails", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { session } = await strictSession([0]);
    warn.mockRestore();

    const [first, second] = await Promise.allSettled([session.run({}), session.run({})]);

    expect(first.status).toBe("rejected");
    expect(second).toEqual({ status: "fulfilled", value: { output0: 1 } });
  });
});

describe("OrtSession.create runtime warm-up", () => {
  it("starts the runtime once per page while a URL model downloads", async () => {
    serveModel(modelProto({ task: "detect" }));

    await OrtSession.create("/models/a.onnx", { providers: ["wasm"] });
    await OrtSession.create("/models/b.onnx", { providers: ["wasm"] });

    expect(calls.filter((call) => call === "probe").length).toBeLessThanOrEqual(1);
    expect(calls.filter((call) => call === "create")).toEqual(["create", "create"]);
  });

  it("does not start a probe session for a model passed as bytes", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await OrtSession.create(modelProto({}), { providers: ["webgpu-never-warmed"] });

    expect(calls).not.toContain("probe");
    vi.restoreAllMocks();
  });
});

describe("OrtSession.create with a quantized model", () => {
  it("pins the metadata key the Python quantize_model writes", async () => {
    const { QUANTIZATION_KEY } = await import("../src/core/session.js");
    expect(QUANTIZATION_KEY).toBe("ort_vision_sdk.quantization");
  });

  it("keeps a quantized model off WebGPU and says why", async () => {
    vi.stubGlobal("navigator", { gpu: { requestAdapter: () => Promise.resolve({}) } });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    serveModel(modelProto({ "ort_vision_sdk.quantization": "qdq-u8s8-per-channel" }));

    const session = await OrtSession.create("/models/q.onnx", { providers: ["webgpu", "wasm"] });

    const call = createSession.mock.calls.at(-1) as unknown[];
    expect((call[1] as Record<string, unknown>).executionProviders).toEqual(["wasm"]);
    expect(session.providers).toEqual(["wasm"]);
    const messages = warn.mock.calls.map((c) => String(c[0]));
    expect(messages.some((m) => m.includes("quantize_model"))).toBe(true);
    expect(messages.some((m) => m.includes("This browser cannot offer"))).toBe(false);
    warn.mockRestore();
  });

  it("falls back to WASM when WebGPU was the only provider asked for", async () => {
    vi.stubGlobal("navigator", { gpu: { requestAdapter: () => Promise.resolve({}) } });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    serveModel(modelProto({ "ort_vision_sdk.quantization": "qdq-u8s8-per-tensor" }));

    const session = await OrtSession.create("/models/q.onnx", { providers: ["webgpu"] });

    expect(session.providers).toEqual(["wasm"]);
    warn.mockRestore();
  });

  it("leaves an unmarked model on WebGPU", async () => {
    vi.stubGlobal("navigator", { gpu: { requestAdapter: () => Promise.resolve({}) } });
    serveModel(modelProto({ task: "detect" }));

    const session = await OrtSession.create("/models/fp32.onnx", { providers: ["webgpu", "wasm"] });

    expect(session.providers).toEqual(["webgpu", "wasm"]);
  });
});
