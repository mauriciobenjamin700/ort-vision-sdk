import { afterEach, describe, expect, it, vi } from "vitest";

/**
 * Tests for treating an empty detection result as an error, on request.
 *
 * Finding nothing is a successful inference — a photo of an empty field is a
 * valid photo — so the SDK returns an empty envelope by default, and every test
 * here that does not opt in asserts exactly that. `raiseOnEmpty` exists for the
 * opposite situation: a step whose precondition is that something is there.
 *
 * ONNX Runtime and `letterbox` are mocked. The first because there is no model,
 * the second because it resamples through a canvas Node does not have — and
 * neither is what these tests are about. Decoding and the threshold decision are
 * the real implementations.
 */

/** Tensors the stubbed session answers a `run` call with. */
let cannedOutputs: Record<string, FakeTensor> = {};

/** A stand-in for `ort.Tensor` carrying just what the SDK reads. */
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
        outputNames: Object.keys(cannedOutputs),
        inputMetadata: undefined,
        outputMetadata: undefined,
        run: () => Promise.resolve(cannedOutputs),
        release: () => Promise.resolve(),
      }),
    ),
  },
}));

vi.mock("../src/preprocess/image.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/preprocess/image.js")>();
  const { RGBImage: Image } = await import("../src/types.js");
  return {
    ...actual,
    letterbox: (_image: unknown, targetWidth: number, targetHeight: number) => ({
      image: new Image(new Uint8Array(targetWidth * targetHeight * 3), targetWidth, targetHeight),
      scale: 1,
      padLeft: 0,
      padTop: 0,
    }),
  };
});

const { NoDetectionsError } = await import("../src/core/exceptions.js");
const { requireDetections } = await import("../src/tasks/base.js");
const { Detector } = await import("../src/tasks/detector.js");
const { Segmenter } = await import("../src/tasks/segmenter.js");
const { RGBImage } = await import("../src/types.js");

/**
 * A blank 64x64 RGB image.
 *
 * @returns The image.
 */
function image(): RGBImage {
  return new RGBImage(new Uint8Array(64 * 64 * 3), 64, 64);
}

/**
 * Install a YOLO head whose single candidate scores `score`.
 *
 * @param score Class score of the one real anchor.
 */
function serveDetectorHead(score: number): void {
  const head = new Float32Array(1 * 6 * 8);
  const anchors = 8;
  head[0 * anchors] = 16;
  head[1 * anchors] = 16;
  head[2 * anchors] = 12;
  head[3 * anchors] = 12;
  head[4 * anchors] = score;
  cannedOutputs = { output0: new FakeTensor("float32", head, [1, 6, 8]) };
}

/**
 * Install a YOLO-seg pair whose single candidate scores `score`.
 *
 * Per-anchor channels are `4 + numClasses + 32`, the layout a seg export
 * produces.
 *
 * @param score Class score of the one real anchor.
 */
function serveSegmenterHead(score: number): void {
  const anchors = 64;
  const perAnchor = new Float32Array(38 * anchors);
  perAnchor[0 * anchors] = 16;
  perAnchor[1 * anchors] = 16;
  perAnchor[2 * anchors] = 12;
  perAnchor[3 * anchors] = 12;
  perAnchor[4 * anchors] = score;
  cannedOutputs = {
    output0: new FakeTensor("float32", perAnchor, [1, 38, anchors]),
    output1: new FakeTensor("float32", new Float32Array(32 * 16 * 16), [1, 32, 16, 16]),
  };
}

afterEach(() => {
  cannedOutputs = {};
});

describe("requireDetections", () => {
  it("stays quiet when the flag is off", () => {
    expect(() =>
      requireDetections(0, {
        raiseOnEmpty: false,
        confThreshold: 0.25,
        classes: undefined,
        path: null,
      }),
    ).not.toThrow();
  });

  it("stays quiet when something was found", () => {
    expect(() =>
      requireDetections(3, {
        raiseOnEmpty: true,
        confThreshold: 0.9,
        classes: undefined,
        path: null,
      }),
    ).not.toThrow();
  });

  it("throws on an empty result", () => {
    expect(() =>
      requireDetections(0, {
        raiseOnEmpty: true,
        confThreshold: 0.25,
        classes: undefined,
        path: null,
      }),
    ).toThrow(NoDetectionsError);
  });

  it("names the threshold that produced the emptiness", () => {
    expect(() =>
      requireDetections(0, {
        raiseOnEmpty: true,
        confThreshold: 0.9,
        classes: undefined,
        path: null,
      }),
    ).toThrow(/confThreshold=0.9/);
  });

  it("names the image when the input was a path", () => {
    expect(() =>
      requireDetections(0, {
        raiseOnEmpty: true,
        confThreshold: 0.25,
        classes: undefined,
        path: "flock.jpg",
      }),
    ).toThrow(/in flock.jpg/);
  });

  it("names the class filter when one narrowed the search", () => {
    expect(() =>
      requireDetections(0, {
        raiseOnEmpty: true,
        confThreshold: 0.25,
        classes: [3, 0],
        path: null,
      }),
    ).toThrow(/among classes \[0, 3\]/);
  });
});

describe("Detector raiseOnEmpty", () => {
  it("returns an empty envelope by default", async () => {
    serveDetectorHead(0.01);
    const detector = await Detector.create(new Uint8Array([]), { labels: ["cat", "dog"] });

    expect((await detector.predict(image()))[0]?.length).toBe(0);
  });

  it("throws when the constructor asked for it", async () => {
    serveDetectorHead(0.01);
    const detector = await Detector.create(new Uint8Array([]), {
      labels: ["cat", "dog"],
      raiseOnEmpty: true,
    });

    await expect(detector.predict(image())).rejects.toThrow(NoDetectionsError);
  });

  it("stays quiet when something was detected", async () => {
    serveDetectorHead(0.9);
    const detector = await Detector.create(new Uint8Array([]), {
      labels: ["cat", "dog"],
      raiseOnEmpty: true,
    });

    expect((await detector.predict(image()))[0]?.length).toBe(1);
  });

  it("per-call override turns it on", async () => {
    serveDetectorHead(0.01);
    const detector = await Detector.create(new Uint8Array([]), { labels: ["cat", "dog"] });

    await expect(detector.predict(image(), { raiseOnEmpty: true })).rejects.toThrow(
      NoDetectionsError,
    );
  });

  it("per-call override turns it off", async () => {
    serveDetectorHead(0.01);
    const detector = await Detector.create(new Uint8Array([]), {
      labels: ["cat", "dog"],
      raiseOnEmpty: true,
    });

    expect((await detector.predict(image(), { raiseOnEmpty: false }))[0]?.length).toBe(0);
  });

  it("reports the stricter per-call threshold that emptied the result", async () => {
    serveDetectorHead(0.5);
    const detector = await Detector.create(new Uint8Array([]), {
      labels: ["cat", "dog"],
      raiseOnEmpty: true,
    });

    await expect(detector.predict(image(), { confThreshold: 0.8 })).rejects.toThrow(
      /confThreshold=0.8/,
    );
  });

  it("throws when a class filter empties the result", async () => {
    serveDetectorHead(0.9);
    const detector = await Detector.create(new Uint8Array([]), {
      labels: ["cat", "dog"],
      raiseOnEmpty: true,
    });

    await expect(detector.predict(image(), { classes: [1] })).rejects.toThrow(
      /among classes \[1\]/,
    );
  });

  it("call() forwards the flag", async () => {
    serveDetectorHead(0.01);
    const detector = await Detector.create(new Uint8Array([]), { labels: ["cat", "dog"] });

    await expect(detector.call(image(), { raiseOnEmpty: true })).rejects.toThrow(
      NoDetectionsError,
    );
  });
});

describe("Segmenter raiseOnEmpty", () => {
  it("returns an empty envelope by default", async () => {
    serveSegmenterHead(0.01);
    const segmenter = await Segmenter.create(new Uint8Array([]), { labels: ["cat", "dog"] });

    expect((await segmenter.predict(image()))[0]?.length).toBe(0);
  });

  it("throws when the constructor asked for it", async () => {
    serveSegmenterHead(0.01);
    const segmenter = await Segmenter.create(new Uint8Array([]), {
      labels: ["cat", "dog"],
      raiseOnEmpty: true,
    });

    await expect(segmenter.predict(image())).rejects.toThrow(NoDetectionsError);
  });

  it("stays quiet when something was detected", async () => {
    serveSegmenterHead(0.9);
    const segmenter = await Segmenter.create(new Uint8Array([]), {
      labels: ["cat", "dog"],
      raiseOnEmpty: true,
    });

    expect((await segmenter.predict(image()))[0]?.length).toBe(1);
  });

  it("per-call override turns it off", async () => {
    serveSegmenterHead(0.01);
    const segmenter = await Segmenter.create(new Uint8Array([]), {
      labels: ["cat", "dog"],
      raiseOnEmpty: true,
    });

    expect((await segmenter.predict(image(), { raiseOnEmpty: false }))[0]?.length).toBe(0);
  });

  it("per-call override turns it on", async () => {
    serveSegmenterHead(0.01);
    const segmenter = await Segmenter.create(new Uint8Array([]), { labels: ["cat", "dog"] });

    await expect(segmenter.predict(image(), { raiseOnEmpty: true })).rejects.toThrow(
      NoDetectionsError,
    );
  });
});
