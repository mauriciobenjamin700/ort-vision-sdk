import { describe, expect, it, vi } from "vitest";

/**
 * Tests for the classifier's session-level behaviour: what it feeds the graph
 * on a warm-up, and that a prediction goes through the fused preprocessing
 * pipeline rather than the composable primitives.
 *
 * ONNX Runtime is mocked — there is no model here — and so is the pipeline's
 * canvas half, which Node has none of. The arithmetic those two stub out is
 * covered against the composable path in `preprocess.pipeline.test.ts`.
 */

/** A stand-in for `ort.Tensor` carrying just what the SDK reads. */
class FakeTensor {
  constructor(
    public readonly type: string,
    public readonly data: Float32Array,
    public readonly dims: number[],
  ) {}
}

/** Feeds the stubbed session was handed, one entry per `run` call. */
let capturedFeeds: Array<Record<string, FakeTensor>> = [];

/** How many times a `ResizePipeline` was asked to preprocess. */
let pipelineRuns = 0;

/** How many `ResizePipeline` instances the task built. */
let pipelinesBuilt = 0;

vi.mock("onnxruntime-web", () => ({
  Tensor: FakeTensor,
  InferenceSession: {
    create: vi.fn(() =>
      Promise.resolve({
        inputNames: ["images"],
        outputNames: ["logits"],
        inputMetadata: undefined,
        outputMetadata: undefined,
        run: (feeds: Record<string, FakeTensor>) => {
          capturedFeeds.push(feeds);
          return Promise.resolve({
            logits: new FakeTensor("float32", new Float32Array([2, 1, 0]), [1, 3]),
          });
        },
        release: () => Promise.resolve(),
      }),
    ),
  },
}));

vi.mock("../src/preprocess/pipeline.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/preprocess/pipeline.js")>();
  return {
    ...actual,
    ResizePipeline: class {
      constructor(
        private readonly w: number,
        private readonly h: number,
      ) {
        pipelinesBuilt += 1;
      }
      run(): { data: Float32Array; reused: boolean } {
        pipelineRuns += 1;
        return { data: new Float32Array(3 * this.w * this.h), reused: true };
      }
      release(): void {}
    },
  };
});

const { Classifier } = await import("../src/tasks/classifier.js");
const { RGBImage } = await import("../src/types.js");

/** A blank 32x32 RGB image. */
function image(): RGBImage {
  return new RGBImage(new Uint8Array(32 * 32 * 3), 32, 32);
}

/** A classifier over a model that declares nothing, so the 224 default applies. */
async function classifier(): Promise<InstanceType<typeof Classifier>> {
  capturedFeeds = [];
  pipelineRuns = 0;
  pipelinesBuilt = 0;
  return Classifier.create(new ArrayBuffer(8), { labels: ["a", "b", "c"] });
}

describe("Classifier warmup", () => {
  it("runs the graph without needing a real image", async () => {
    const clf = await classifier();

    await clf.warmup();

    expect(capturedFeeds).toHaveLength(1);
    expect(capturedFeeds[0]?.images?.dims).toEqual([1, 3, 224, 224]);
    expect(capturedFeeds[0]?.images?.data.every((value) => value === 0)).toBe(true);
  });

  it("does not build the preprocessing pipeline", async () => {
    const clf = await classifier();

    await clf.warmup();

    expect(pipelinesBuilt).toBe(0);
  });

  it("honours the requested number of runs", async () => {
    const clf = await classifier();

    await clf.warmup(3);

    expect(capturedFeeds).toHaveLength(3);
  });
});

describe("Classifier preprocessing", () => {
  it("goes through the fused pipeline", async () => {
    const clf = await classifier();

    await clf.predict(image());

    expect(pipelineRuns).toBe(1);
    expect(capturedFeeds[0]?.images?.dims).toEqual([1, 3, 224, 224]);
  });

  it("builds the pipeline once and reuses it across predictions", async () => {
    const clf = await classifier();

    await clf.predict(image());
    await clf.predict(image());

    expect(pipelinesBuilt).toBe(1);
    expect(pipelineRuns).toBe(2);
  });
});
