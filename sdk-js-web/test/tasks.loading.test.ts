import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * How the tasks consume a decoded input — and what they are careful not to do.
 *
 * Three decisions are pinned here, each one a performance change that would be
 * a correctness bug if it went one step further:
 *
 * - The canvas `loadImageSource` decoded onto is handed to the letterbox
 *   pipeline, so it can draw from it instead of rebuilding RGBA.
 * - `croppedImage` / `segmentedImage` are built on first read, but only when
 *   the SDK owns the pixel buffer. An `RGBImage` the caller passed is theirs to
 *   refill, so it is cropped up front.
 * - `Segmenter.create` subtracts the mask coefficients when it reads the class
 *   count off the graph. Every other test mocks ORT with no output metadata,
 *   which is how inferring 33 classes for a one-class model went unnoticed.
 *
 * ONNX Runtime, the letterbox pipeline and (for the ownership cases) the image
 * loader are mocked; decoding is real.
 */

class FakeTensor {
  constructor(
    public readonly type: string,
    public readonly data: Float32Array,
    public readonly dims: number[],
  ) {}
}

/** Tensors the stubbed session answers a `run` call with. */
let cannedOutputs: Record<string, FakeTensor> = {};

/** Declared output metadata the stubbed session reports. */
let outputMetadata: unknown = undefined;

/** Arguments every stubbed `LetterboxPipeline.run` call received. */
const pipelineCalls: unknown[][] = [];

/** What the mocked loader returns, or `null` to use the real one. */
let loaded: { image: unknown; canvas: unknown; owned: boolean } | null = null;

vi.mock("onnxruntime-web", () => ({
  Tensor: FakeTensor,
  InferenceSession: {
    create: vi.fn(() =>
      Promise.resolve({
        inputNames: ["images"],
        get outputNames() {
          return Object.keys(cannedOutputs);
        },
        inputMetadata: undefined,
        get outputMetadata() {
          return outputMetadata;
        },
        run: () => Promise.resolve(cannedOutputs),
        release: () => Promise.resolve(),
      }),
    ),
  },
}));

vi.mock("../src/preprocess/pipeline.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/preprocess/pipeline.js")>();
  return {
    ...actual,
    LetterboxPipeline: class {
      constructor(
        private readonly w: number,
        private readonly h: number,
      ) {}
      run(...args: unknown[]) {
        pipelineCalls.push(args);
        return {
          data: new Float32Array(3 * this.w * this.h),
          scale: 1,
          padLeft: 0,
          padTop: 0,
          reused: false,
        };
      }
      release(): void {}
    },
  };
});

vi.mock("../src/io/image.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/io/image.js")>();
  return {
    ...actual,
    loadImageSource: (source: unknown) =>
      loaded !== null ? Promise.resolve(loaded) : actual.loadImageSource(source as never),
  };
});

const { Detector } = await import("../src/tasks/detector.js");
const { Segmenter } = await import("../src/tasks/segmenter.js");
const { cropToBox, memoize } = await import("../src/tasks/base.js");
const { BoundingBox, RGBImage } = await import("../src/types.js");

beforeEach(() => {
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
  cannedOutputs = {};
  outputMetadata = undefined;
  pipelineCalls.length = 0;
  loaded = null;
});

/**
 * A 64x64 image whose every byte encodes its own position, so a crop of the
 * wrong region cannot pass by accident.
 */
function gradient(): RGBImage {
  const data = new Uint8Array(64 * 64 * 3);
  for (let i = 0; i < data.length; i++) data[i] = i % 251;
  return new RGBImage(data, 64, 64);
}

/** Install a detect head with one candidate box at (10, 10)-(30, 30). */
function serveDetectorHead(): void {
  const anchors = 8;
  const head = new Float32Array(6 * anchors);
  head[0] = 20;
  head[anchors] = 20;
  head[2 * anchors] = 20;
  head[3 * anchors] = 20;
  head[4 * anchors] = 0.9;
  cannedOutputs = { output0: new FakeTensor("float32", head, [1, 6, anchors]) };
}

describe("Segmenter.create class count", () => {
  it("does not count mask coefficients as classes", async () => {
    outputMetadata = [
      { name: "output0", isTensor: true, type: "float32", shape: [1, 37, 8400] },
      { name: "output1", isTensor: true, type: "float32", shape: [1, 32, 160, 160] },
    ];

    const segmenter = await Segmenter.create(new Uint8Array(0), { readMetadata: false });

    expect(segmenter.numClasses).toBe(1);
  });
});

describe("Detector crops", () => {
  it("hands the decoded canvas to the letterbox pipeline", async () => {
    serveDetectorHead();
    const canvas = { tag: "decoded-canvas" };
    loaded = { image: gradient(), canvas, owned: true };
    const detector = await Detector.create(new Uint8Array(0), { readMetadata: false });

    await detector.predict("/frame.png");

    expect(pipelineCalls[0]?.[1]).toBe(canvas);
  });

  it("defers the crop when the SDK owns the pixels", async () => {
    serveDetectorHead();
    const frame = gradient();
    loaded = { image: frame, canvas: null, owned: true };
    const detector = await Detector.create(new Uint8Array(0), { readMetadata: false });

    const detection = [...(await detector.predict("/frame.png"))[0]!][0]!;
    frame.data.fill(7);

    expect(detection.croppedImage.data.every((v) => v === 7)).toBe(true);
    expect(detection.croppedImage).toBe(detection.croppedImage);
  });

  it("crops up front when the caller passed the RGBImage", async () => {
    serveDetectorHead();
    const frame = gradient();
    const expected = cropToBox(frame, new BoundingBox(10, 10, 30, 30));
    const detector = await Detector.create(new Uint8Array(0), { readMetadata: false });

    const detection = [...(await detector.predict(frame))[0]!][0]!;
    frame.data.fill(7);

    expect(detection.croppedImage.data).toEqual(expected.data);
  });
});

describe("memoize", () => {
  it("computes once and returns the same value after", () => {
    const compute = vi.fn(() => ({}));
    const get = memoize(compute);

    expect(get()).toBe(get());
    expect(compute).toHaveBeenCalledTimes(1);
  });
});

describe("cropToBox", () => {
  it("clamps the box to the image and copies the covered rows", () => {
    const frame = gradient();

    const crop = cropToBox(frame, new BoundingBox(-5, 62, 3, 80));

    expect([crop.width, crop.height]).toEqual([3, 2]);
    expect(Array.from(crop.data.subarray(0, 9))).toEqual(
      Array.from(frame.data.subarray(62 * 64 * 3, 62 * 64 * 3 + 9)),
    );
  });

  it("returns an empty image for a box with no area", () => {
    const crop = cropToBox(gradient(), new BoundingBox(10, 10, 10, 30));

    expect([crop.width, crop.height, crop.data.length]).toEqual([0, 0, 0]);
  });
});
