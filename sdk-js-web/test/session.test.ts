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

const createSession = vi.fn((model: unknown, _options?: unknown) => {
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

    expect(calls).toEqual(["metadata", "create"]);
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
