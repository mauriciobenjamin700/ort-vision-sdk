import { describe, expect, it, vi } from "vitest";

/**
 * Tests for picking the preprocessing a classifier was actually trained with.
 *
 * The decision is invisible from the outside — a classifier fed the wrong tensor
 * returns a prediction of exactly the right shape, just a worse one — so the
 * task-level tests read the `mean`/`std` handed to the preprocessing pipeline
 * rather than the prediction that comes back.
 *
 * Mirrors `tests/test_normalization.py` in the Python SDK.
 */

const ULTRALYTICS: Record<string, string> = {
  author: "Ultralytics",
  task: "classify",
  names: "{0: 'a', 1: 'b'}",
};
const TORCHVISION: Record<string, string> = {};

/** `[mean, std]` of every `ResizePipeline` the task built. */
let pipelineArgs: Array<[readonly number[], readonly number[]]> = [];

/** Metadata the mocked session reports. */
let sessionMetadata: Record<string, string> = {};

class FakeTensor {
  constructor(
    public readonly type: string,
    public readonly data: Float32Array,
    public readonly dims: number[],
  ) {}
}

vi.mock("onnxruntime-web", () => ({
  Tensor: FakeTensor,
  InferenceSession: {
    create: vi.fn(() =>
      Promise.resolve({
        inputNames: ["images"],
        outputNames: ["logits"],
        inputMetadata: undefined,
        outputMetadata: undefined,
        run: () =>
          Promise.resolve({
            logits: new FakeTensor("float32", new Float32Array([2, 1]), [1, 2]),
          }),
        release: () => Promise.resolve(),
      }),
    ),
  },
}));

vi.mock("../src/core/metadata.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/core/metadata.js")>();
  return { ...actual, readModelMetadata: () => sessionMetadata };
});

vi.mock("../src/preprocess/pipeline.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/preprocess/pipeline.js")>();
  return {
    ...actual,
    ResizePipeline: class {
      constructor(
        private readonly w: number,
        private readonly h: number,
        mean: readonly number[],
        std: readonly number[],
      ) {
        pipelineArgs.push([mean, std]);
      }
      run(): { data: Float32Array; reused: boolean } {
        return { data: new Float32Array(3 * this.w * this.h), reused: true };
      }
      release(): void {}
    },
  };
});

const {
  IDENTITY_MEAN,
  IDENTITY_STD,
  IMAGENET_MEAN,
  IMAGENET_STD,
  isUltralyticsClassifier,
  resolveNormalization,
} = await import("../src/normalization.js");
const { Classifier } = await import("../src/tasks/classifier.js");
const { RGBImage } = await import("../src/types.js");

describe("isUltralyticsClassifier", () => {
  it("recognizes an Ultralytics classification export", () => {
    expect(isUltralyticsClassifier(ULTRALYTICS)).toBe(true);
  });

  it("is case and whitespace insensitive", () => {
    expect(isUltralyticsClassifier({ author: " ultralytics ", task: "Classify" })).toBe(true);
  });

  it("rejects an Ultralytics detector", () => {
    expect(isUltralyticsClassifier({ author: "Ultralytics", task: "detect" })).toBe(false);
  });

  it("rejects a model with no metadata", () => {
    expect(isUltralyticsClassifier({})).toBe(false);
  });
});

describe("resolveNormalization", () => {
  it("picks the identity for an Ultralytics export", () => {
    expect(resolveNormalization(ULTRALYTICS)).toEqual({
      name: "ultralytics",
      mean: IDENTITY_MEAN,
      std: IDENTITY_STD,
    });
  });

  it("picks ImageNet for everything else", () => {
    expect(resolveNormalization(TORCHVISION)).toEqual({
      name: "imagenet",
      mean: IMAGENET_MEAN,
      std: IMAGENET_STD,
    });
  });

  it("treats 'none' as the same arithmetic as 'ultralytics'", () => {
    const asNone = resolveNormalization(TORCHVISION, { normalization: "none" });
    const asVendor = resolveNormalization(TORCHVISION, { normalization: "ultralytics" });

    expect(asNone.mean).toEqual(asVendor.mean);
    expect(asNone.std).toEqual(asVendor.std);
    expect([asNone.name, asVendor.name]).toEqual(["none", "ultralytics"]);
  });

  it("keeps the preset deviation when only a mean is given", () => {
    expect(resolveNormalization(TORCHVISION, { mean: [0, 0, 0] })).toEqual({
      name: "custom",
      mean: [0, 0, 0],
      std: IMAGENET_STD,
    });
  });

  it("rejects a preset alongside explicit values", () => {
    expect(() =>
      resolveNormalization(TORCHVISION, { normalization: "imagenet", mean: [0, 0, 0] }),
    ).toThrow(RangeError);
  });

  it("rejects an unknown preset", () => {
    expect(() =>
      resolveNormalization(TORCHVISION, {
        normalization: "torchvision" as never,
      }),
    ).toThrow(/normalization must be one of/);
  });

  it("warns when an Ultralytics model is normalized", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    resolveNormalization(ULTRALYTICS, { mean: IMAGENET_MEAN, std: IMAGENET_STD });

    expect(warn).toHaveBeenCalledOnce();
    expect(warn.mock.calls[0]?.[0]).toMatch(/Ultralytics export/);
    warn.mockRestore();
  });

  it("stays quiet when the override is the identity", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    resolveNormalization(ULTRALYTICS, { mean: IDENTITY_MEAN, std: IDENTITY_STD });

    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });
});

describe("Classifier preprocessing", () => {
  /**
   * Build a classifier over a model whose metadata is `metadata`.
   *
   * @param metadata Metadata the mocked session reports.
   * @param options Extra classifier options.
   */
  async function build(
    metadata: Record<string, string>,
    options: Record<string, unknown> = {},
  ): Promise<InstanceType<typeof Classifier>> {
    pipelineArgs = [];
    sessionMetadata = metadata;
    return Classifier.create(new ArrayBuffer(8), { labels: ["a", "b"], ...options });
  }

  it("feeds an Ultralytics model raw values", async () => {
    const clf = await build(ULTRALYTICS);

    await clf.predict(new RGBImage(new Uint8Array(32 * 32 * 3), 32, 32));

    expect(clf.normalization).toBe("ultralytics");
    expect(pipelineArgs[0]).toEqual([IDENTITY_MEAN, IDENTITY_STD]);
  });

  it("feeds anything else ImageNet-normalized values", async () => {
    const clf = await build(TORCHVISION);

    await clf.predict(new RGBImage(new Uint8Array(32 * 32 * 3), 32, 32));

    expect(clf.normalization).toBe("imagenet");
    expect(pipelineArgs[0]).toEqual([IMAGENET_MEAN, IMAGENET_STD]);
  });

  it("lets an explicit preset override the detection", async () => {
    const clf = await build(ULTRALYTICS, { normalization: "imagenet" });

    await clf.predict(new RGBImage(new Uint8Array(32 * 32 * 3), 32, 32));

    expect(pipelineArgs[0]).toEqual([IMAGENET_MEAN, IMAGENET_STD]);
  });

  it("lets explicit values win", async () => {
    const clf = await build(TORCHVISION, { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] });

    await clf.predict(new RGBImage(new Uint8Array(32 * 32 * 3), 32, 32));

    expect(clf.normalization).toBe("custom");
    expect(pipelineArgs[0]).toEqual([
      [0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5],
    ]);
  });
});
