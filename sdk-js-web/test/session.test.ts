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

const createSession = vi.fn((model: unknown) => {
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
