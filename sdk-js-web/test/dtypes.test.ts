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
 *
 * `Float16Array` is not a constant of the environment — it reached V8 in
 * Node 24, and the CI matrix runs 18, 20 and 22 — which is the whole reason
 * this SDK refuses such a model up front. So the tests never assume it is
 * there: the half-precision path is exercised against whichever constructor
 * exists (the native one where it does, a stand-in where it does not) and the
 * refusal path is exercised with the global removed. Both run on every Node.
 */

import { readFileSync } from "node:fs";

import { afterEach, describe, expect, it, vi } from "vitest";

import { readModelInputTypes } from "../src/core/metadata.js";

const MODELS = new URL("../../sdk-python/tests/fixtures/models/", import.meta.url);

/** Read one of the shared fixture models into memory. */
function model(name: string): Uint8Array {
  return new Uint8Array(readFileSync(new URL(name, MODELS)));
}

/**
 * A `Float16Array` stand-in for Node releases that have none.
 *
 * Deliberately not a `Float32Array` subclass: the assertions below check that a
 * converted feed is *not* a `Float32Array`, which a subclass would satisfy and
 * so would pass while proving nothing. Values are stored as given — this stands
 * in for the constructor, not for half-precision rounding, and every value the
 * tests use is exactly representable in float16 anyway.
 */
class StubFloat16Array {
  readonly length: number;
  readonly buffer: ArrayBuffer = new ArrayBuffer(0);
  [index: number]: number;

  constructor(input: ArrayLike<number> | number) {
    const values = typeof input === "number" ? new Array<number>(input).fill(0) : Array.from(input);
    this.length = values.length;
    values.forEach((value, index) => {
      this[index] = value;
    });
  }
}

/** Whether this Node release ships the real constructor. */
const NATIVE_FLOAT16 = "Float16Array" in globalThis;

/**
 * Import the dtype helpers with a usable `Float16Array` in place.
 *
 * The module reads the global once, at load, so the stub has to be installed
 * before the import rather than before the call.
 *
 * @returns The freshly imported module.
 */
async function withFloat16(): Promise<typeof import("../src/core/dtypes.js")> {
  if (!NATIVE_FLOAT16) vi.stubGlobal("Float16Array", StubFloat16Array);
  vi.resetModules();
  return import("../src/core/dtypes.js");
}

/**
 * Import the dtype helpers with `Float16Array` removed.
 *
 * @returns The freshly imported module.
 */
async function withoutFloat16(): Promise<typeof import("../src/core/dtypes.js")> {
  vi.stubGlobal("Float16Array", undefined);
  vi.resetModules();
  return import("../src/core/dtypes.js");
}

afterEach(() => {
  vi.resetModules();
  vi.unstubAllGlobals();
});

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
  it("names the two types an export realistically declares", async () => {
    const { tensorTypeFor } = await withFloat16();

    expect(tensorTypeFor(1)).toBe("float32");
    expect(tensorTypeFor(10)).toBe("float16");
  });

  it("falls back to float32 for an unknown or absent type", async () => {
    const { tensorTypeFor } = await withFloat16();

    expect(tensorTypeFor(undefined)).toBe("float32");
    expect(tensorTypeFor(999)).toBe("float32");
  });
});

describe("toFeedData", () => {
  it("hands a float32 feed straight through", async () => {
    const { toFeedData } = await withFloat16();
    const data = new Float32Array([0.25, 0.5]);

    expect(toFeedData(data, "float32")).toBe(data);
  });

  it("converts to a half-precision array for a half-precision input", async () => {
    const { toFeedData } = await withFloat16();

    const converted = toFeedData(new Float32Array([0.25, 0.5]), "float16");

    expect(converted.length).toBe(2);
    expect(converted[0]).toBe(0.25);
    expect(converted[1]).toBe(0.5);
    expect(converted).not.toBeInstanceOf(Float32Array);
  });

  it("refuses to build one where the runtime has no Float16Array", async () => {
    const { toFeedData } = await withoutFloat16();

    expect(() => toFeedData(new Float32Array([0.25]), "float16")).toThrow(RangeError);
  });

  it("still hands float32 through where the runtime has no Float16Array", async () => {
    const { toFeedData } = await withoutFloat16();
    const data = new Float32Array([0.25]);

    expect(toFeedData(data, "float32")).toBe(data);
  });
});

describe("hasFloat16Array", () => {
  it("reports true where the constructor exists", async () => {
    const { hasFloat16Array } = await withFloat16();

    expect(hasFloat16Array()).toBe(true);
  });

  it("reports false where it does not", async () => {
    const { hasFloat16Array } = await withoutFloat16();

    expect(hasFloat16Array()).toBe(false);
  });
});

describe("asFloat32Array", () => {
  it("passes a float32 output through without copying", async () => {
    const { asFloat32Array } = await withFloat16();
    const data = new Float32Array([1, 2, 3]);

    expect(asFloat32Array(data)).toBe(data);
  });

  it("widens a half-precision output before decoding", async () => {
    const { asFloat32Array, toFeedData } = await withFloat16();
    const half = toFeedData(new Float32Array([640.5, 0.25]), "float16");

    const widened = asFloat32Array(half);

    expect(widened).toBeInstanceOf(Float32Array);
    expect(Array.from(widened)).toEqual([640.5, 0.25]);
  });
});

describe("OrtSession with a half-precision graph", () => {
  /**
   * Build the SDK against a stubbed ORT that records what it is run with.
   *
   * @param float16 Whether the environment should be able to build a half
   *   tensor. `false` removes the global, which is the state every Node before
   *   24 and several browsers are actually in.
   * @returns The freshly imported `OrtSession` class plus the recorded feeds.
   */
  async function withStubbedOrt(float16: boolean = true): Promise<{
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
    if (!float16) {
      vi.stubGlobal("Float16Array", undefined);
    } else if (!NATIVE_FLOAT16) {
      vi.stubGlobal("Float16Array", StubFloat16Array);
    }
    vi.resetModules();
    const { OrtSession } = await import("../src/core/session.js");
    return { OrtSession, feeds };
  }

  afterEach(() => {
    vi.doUnmock("onnxruntime-web");
  });

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
    const { OrtSession } = await withStubbedOrt(false);
    const { ModelLoadError } = await import("../src/core/exceptions.js");

    const attempt = OrtSession.create(model("tiny_classifier_fp16.onnx"));

    await expect(attempt).rejects.toThrow(ModelLoadError);
    await expect(attempt).rejects.toThrow(/no Float16Array/);
  });

  it("still loads a float32 model where Float16Array is missing", async () => {
    const { OrtSession } = await withStubbedOrt(false);

    const session = await OrtSession.create(model("tiny_classifier.onnx"));

    expect(session.inputDtype).toBe("float32");
  });
});
