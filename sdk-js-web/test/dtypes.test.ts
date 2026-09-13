/**
 * Half-precision models must be reachable through the public API.
 *
 * A model exported with `half=True` used to load fine and die on the first
 * `predict()` with `Unexpected input data type. Actual: (tensor(float)) ,
 * expected: (tensor(float16))`, because every task built its feed as
 * `Float32Array` and nothing read the type the graph declares. These tests pin
 * the three pieces that close that: reading the declared type off the file,
 * converting the feed at the run boundary, and refusing at `create()` where the
 * runtime has no `Float16Array` to convert into.
 *
 * The models under test are the same `.onnx` fixtures the Python suite loads,
 * which is what makes "both SDKs read the same file the same way" an assertion
 * rather than a hope.
 */

import { readFileSync } from "node:fs";

import { afterEach, describe, expect, it, vi } from "vitest";

import { asFloat32Array, hasFloat16Array, tensorTypeFor, toFeedData } from "../src/core/dtypes.js";
import { readModelInputTypes } from "../src/core/metadata.js";

const MODELS = new URL("../../sdk-python/tests/fixtures/models/", import.meta.url);

/** Read one of the shared fixture models into memory. */
function model(name: string): Uint8Array {
  return new Uint8Array(readFileSync(new URL(name, MODELS)));
}

describe("readModelInputTypes", () => {
  it("reads FLOAT16 off a half-precision export", () => {
    expect(readModelInputTypes(model("tiny_classifier_fp16.onnx"))).toEqual({ images: 10 });
  });

  it("reads FLOAT off a single-precision export", () => {
    expect(readModelInputTypes(model("tiny_classifier.onnx"))).toEqual({ images: 1 });
  });

  it("names every input of a multi-input graph", () => {
    const types = readModelInputTypes(model("tiny_detector.onnx"));

    expect(Object.keys(types)).toEqual(["images"]);
  });

  it("returns nothing for bytes that are not a model", () => {
    expect(readModelInputTypes(new Uint8Array([0, 1, 2, 3]))).toEqual({});
  });
});

describe("tensorTypeFor", () => {
  it("names the two types an export realistically declares", () => {
    expect(tensorTypeFor(1)).toBe("float32");
    expect(tensorTypeFor(10)).toBe("float16");
  });

  it("falls back to float32 for an unknown or absent type", () => {
    expect(tensorTypeFor(undefined)).toBe("float32");
    expect(tensorTypeFor(999)).toBe("float32");
  });
});

describe("toFeedData", () => {
  it("hands a float32 feed straight through", () => {
    const data = new Float32Array([0.25, 0.5]);

    expect(toFeedData(data, "float32")).toBe(data);
  });

  it("converts to Float16Array for a half-precision input", () => {
    const converted = toFeedData(new Float32Array([0.25, 0.5]), "float16");

    expect(converted.length).toBe(2);
    expect(converted[0]).toBe(0.25);
    expect(converted[1]).toBe(0.5);
    expect(converted).not.toBeInstanceOf(Float32Array);
  });

  it("reports that this environment can build half tensors at all", () => {
    expect(hasFloat16Array()).toBe(true);
  });
});

describe("asFloat32Array", () => {
  it("passes a float32 output through without copying", () => {
    const data = new Float32Array([1, 2, 3]);

    expect(asFloat32Array(data)).toBe(data);
  });

  it("widens a half-precision output before decoding", () => {
    const half = toFeedData(new Float32Array([640.5, 0.25]), "float16");

    const widened = asFloat32Array(half);

    expect(widened).toBeInstanceOf(Float32Array);
    expect(Array.from(widened)).toEqual([640.5, 0.25]);
  });
});

describe("OrtSession with a half-precision graph", () => {
  afterEach(() => {
    vi.resetModules();
    vi.unstubAllGlobals();
    vi.doUnmock("onnxruntime-web");
  });

  /**
   * Build the SDK against a stubbed ORT that records what it is run with.
   *
   * @returns The freshly imported `OrtSession` class plus the recorded feeds.
   */
  async function withStubbedOrt(): Promise<{
    OrtSession: typeof import("../src/core/session.js").OrtSession;
    feeds: Record<string, { type: string; data: unknown }>[];
  }> {
    const feeds: Record<string, { type: string; data: unknown }>[] = [];
    vi.doMock("onnxruntime-web", () => ({
      InferenceSession: {
        create: () =>
          Promise.resolve({
            inputNames: ["images"],
            outputNames: ["logits"],
            inputMetadata: undefined,
            outputMetadata: undefined,
            run: (given: Record<string, { type: string; data: unknown }>) => {
              feeds.push(given);
              return Promise.resolve({ logits: { type: "float16", data: new Float32Array([1]) } });
            },
            release: () => Promise.resolve(),
          }),
      },
      Tensor: class {
        constructor(
          public readonly type: string,
          public readonly data: unknown,
          public readonly dims: number[],
        ) {}
      },
    }));
    vi.resetModules();
    const { OrtSession } = await import("../src/core/session.js");
    return { OrtSession, feeds };
  }

  it("reports the type the graph declares", async () => {
    const { OrtSession } = await withStubbedOrt();

    const session = await OrtSession.create(model("tiny_classifier_fp16.onnx"));

    expect(session.inputDtype).toBe("float16");
    expect(session.inputDtypes).toEqual(["float16"]);
  });

  it("reports float32 for a single-precision graph", async () => {
    const { OrtSession } = await withStubbedOrt();

    const session = await OrtSession.create(model("tiny_classifier.onnx"));

    expect(session.inputDtype).toBe("float32");
  });

  it("converts a float32 feed to the declared half precision", async () => {
    const { OrtSession, feeds } = await withStubbedOrt();
    const session = await OrtSession.create(model("tiny_classifier_fp16.onnx"));

    await session.run({
      images: { type: "float32", data: new Float32Array([0.5]), dims: [1] } as never,
    });

    expect(feeds[0]!.images!.type).toBe("float16");
    expect(feeds[0]!.images!.data).not.toBeInstanceOf(Float32Array);
  });

  it("leaves a feed alone when the graph wants float32", async () => {
    const { OrtSession, feeds } = await withStubbedOrt();
    const session = await OrtSession.create(model("tiny_classifier.onnx"));
    const tensor = { type: "float32", data: new Float32Array([0.5]), dims: [1] };

    await session.run({ images: tensor as never });

    expect(feeds[0]!.images).toBe(tensor);
  });

  it("refuses the model at create() when the runtime has no Float16Array", async () => {
    await withStubbedOrt();
    vi.stubGlobal("Float16Array", undefined);
    vi.resetModules();
    const { OrtSession } = await import("../src/core/session.js");
    const { ModelLoadError } = await import("../src/core/exceptions.js");

    const attempt = OrtSession.create(model("tiny_classifier_fp16.onnx"));

    await expect(attempt).rejects.toThrow(ModelLoadError);
    await expect(attempt).rejects.toThrow(/no Float16Array/);
  });

  it("still loads a float32 model where Float16Array is missing", async () => {
    await withStubbedOrt();
    vi.stubGlobal("Float16Array", undefined);
    vi.resetModules();
    const { OrtSession } = await import("../src/core/session.js");

    const session = await OrtSession.create(model("tiny_classifier.onnx"));

    expect(session.inputDtype).toBe("float32");
  });
});
