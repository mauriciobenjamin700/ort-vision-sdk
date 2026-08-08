import { afterEach, describe, expect, it, vi } from "vitest";

/**
 * Tests for the browser-side fused-pipeline runtime.
 *
 * ONNX Runtime is mocked, so these are about the code *around* the graph: which
 * feeds get built, how the graph's letterboxed boxes are mapped back onto the
 * caller's image, which rows are read, and how the two label spaces are kept
 * apart. The graph itself is covered end to end on the Python side, against a
 * real fused file — there is nothing browser-specific about what it computes.
 *
 * One thing here *is* browser-specific and worth the mock: ORT Web hands back
 * `int64` outputs as `BigInt64Array`, whose values neither compare nor index
 * like numbers. The stub returns real `BigInt64Array`s so that conversion is
 * genuinely exercised rather than assumed.
 */

/** Field number of `metadata_props` in `ModelProto`. */
const METADATA_PROPS_FIELD = 14;

/** Tensors the stubbed session answers a `run` call with. */
let cannedOutputs: Record<string, FakeTensor> = {};

/** Feeds the stubbed session was handed on the last `run` call. */
let capturedFeeds: Record<string, FakeTensor> = {};

/** Declared output metadata the stubbed session reports. */
let outputMetadata: unknown = undefined;

/**
 * What the stubbed `letterbox` reports back.
 *
 * The real one resamples through a canvas, which Node has none of, and its
 * arithmetic is covered by its own tests. Stubbing it keeps these tests on the
 * question they exist to answer — that the pipeline feeds the scale and padding
 * it was given, and inverts exactly those when mapping boxes back.
 */
let cannedLetterbox = { scale: 1, padLeft: 0, padTop: 0 };

/** A stand-in for `ort.Tensor` carrying just what the SDK reads. */
class FakeTensor {
  constructor(
    public readonly type: string,
    public readonly data: Float32Array | BigInt64Array,
    public readonly dims: number[],
  ) {}
}

vi.mock("onnxruntime-web", () => ({
  Tensor: FakeTensor,
  InferenceSession: {
    create: vi.fn(() =>
      Promise.resolve({
        inputNames: ["images", "source_image", "letterbox_scale", "letterbox_pad"],
        outputNames: ["boxes", "scores", "classes", "num_detections", "probs"],
        inputMetadata: undefined,
        get outputMetadata() {
          return outputMetadata;
        },
        run: (feeds: Record<string, FakeTensor>) => {
          capturedFeeds = feeds;
          return Promise.resolve(cannedOutputs);
        },
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
      ...cannedLetterbox,
    }),
  };
});

const { FusionError, NoDetectionsError } = await import("../src/core/exceptions.js");
const { METADATA_PREFIX } = await import("../src/fusion.js");
const { DetectClassify } = await import("../src/tasks/detectClassify.js");
const { RGBImage } = await import("../src/types.js");

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
 * Build a `ModelProto` carrying nothing but a fused pipeline's metadata.
 *
 * @param overrides `ovs.*` entries to replace, keys given without the prefix.
 * @returns The encoded model bytes.
 */
function pipelineModel(overrides: Readonly<Record<string, string>> = {}): Uint8Array {
  const spec: Record<string, string> = {
    kind: "detect_classify",
    sdk_version: "0.0.0",
    input_size: "64,64",
    crop_size: "8,8",
    crop_source: "detector_input",
    max_detections: "4",
    conf_threshold: "0.25",
    iou_threshold: "0.45",
    apply_softmax: "1",
    detector_names: "{0: 'cat', 1: 'dog'}",
    classifier_names: "{0: 'red', 1: 'green', 2: 'blue'}",
    ...overrides,
  };
  const encoder = new TextEncoder();
  const bytes: number[] = [];
  for (const [key, value] of Object.entries(spec)) {
    const entry = [
      ...lengthDelimited(1, [...encoder.encode(`${METADATA_PREFIX}${key}`)]),
      ...lengthDelimited(2, [...encoder.encode(value)]),
    ];
    bytes.push(...lengthDelimited(METADATA_PROPS_FIELD, entry));
  }
  return new Uint8Array(bytes);
}

/**
 * Install a canned output set with four rows, two of them real.
 *
 * @param overrides Per-output replacements.
 */
function serveOutputs(
  overrides: {
    boxes?: number[];
    scores?: number[];
    classes?: bigint[];
    count?: number;
    probs?: number[];
  } = {},
): void {
  const boxes = overrides.boxes ?? [8, 8, 24, 24, 40, 40, 56, 56, 0, 0, 0, 0, 0, 0, 0, 0];
  const scores = overrides.scores ?? [0.9, 0.4, 0, 0];
  const classes = overrides.classes ?? [0n, 1n, 0n, 0n];
  const probs = overrides.probs ?? [3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0];
  cannedOutputs = {
    boxes: new FakeTensor("float32", new Float32Array(boxes), [4, 4]),
    scores: new FakeTensor("float32", new Float32Array(scores), [4]),
    classes: new FakeTensor("int64", BigInt64Array.from(classes), [4]),
    num_detections: new FakeTensor(
      "int64",
      BigInt64Array.from([BigInt(overrides.count ?? 2)]),
      [1],
    ),
    probs: new FakeTensor("float32", new Float32Array(probs), [4, 3]),
  };
}

/**
 * A blank square RGB image.
 *
 * @param size Side length in pixels.
 * @returns The image.
 */
function image(size = 64): RGBImage {
  return new RGBImage(new Uint8Array(size * size * 3), size, size);
}

afterEach(() => {
  cannedOutputs = {};
  capturedFeeds = {};
  outputMetadata = undefined;
  cannedLetterbox = { scale: 1, padLeft: 0, padTop: 0 };
});

describe("DetectClassify.create", () => {
  it("rejects a model that carries no pipeline metadata", async () => {
    await expect(DetectClassify.create(new Uint8Array([]))).rejects.toThrow(FusionError);
  });

  it("reads the spec out of the model", async () => {
    const pipeline = await DetectClassify.create(pipelineModel());

    expect(pipeline.inputSize).toEqual([64, 64]);
    expect(pipeline.spec.cropSize).toEqual([8, 8]);
  });

  it("exposes both label spaces separately", async () => {
    const pipeline = await DetectClassify.create(pipelineModel());

    expect(pipeline.names).toEqual({ 0: "cat", 1: "dog" });
    expect(pipeline.classifierNames).toEqual({ 0: "red", 1: "green", 2: "blue" });
  });

  it("falls back to COCO for an unnamed detection stage", async () => {
    const model = pipelineModel({ detector_names: "" });
    const pipeline = await DetectClassify.create(model);

    expect(pipeline.labels).toHaveLength(80);
    expect(pipeline.labels[0]).toBe("person");
  });

  it("generates names for an unnamed classification stage from the graph's shape", async () => {
    outputMetadata = [
      { name: "boxes", isTensor: true, shape: [4, 4] },
      { name: "scores", isTensor: true, shape: [4] },
      { name: "classes", isTensor: true, shape: [4] },
      { name: "num_detections", isTensor: true, shape: [1] },
      { name: "probs", isTensor: true, shape: [4, 3] },
    ];
    const pipeline = await DetectClassify.create(pipelineModel({ classifier_names: "" }));

    expect(pipeline.classifierLabels).toEqual(["class_0", "class_1", "class_2"]);
  });

  it("lets caller labels win over the recorded ones", async () => {
    const pipeline = await DetectClassify.create(pipelineModel(), {
      labels: ["sheep", "goat"],
      classifierLabels: ["a", "b", "c"],
    });

    expect(pipeline.names).toEqual({ 0: "sheep", 1: "goat" });
    expect(pipeline.classifierNames).toEqual({ 0: "a", 1: "b", 2: "c" });
  });
});

describe("DetectClassify feeds", () => {
  it("sends only the letterboxed image by default", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    await pipeline.predict(image());

    expect(Object.keys(capturedFeeds)).toEqual(["images"]);
    expect(capturedFeeds.images?.dims).toEqual([1, 3, 64, 64]);
  });

  it("sends the source image and its letterbox for the original crop source", async () => {
    serveOutputs();
    cannedLetterbox = { scale: 0.5, padLeft: 0, padTop: 0 };
    const pipeline = await DetectClassify.create(
      pipelineModel({ crop_source: "original" }),
    );
    await pipeline.predict(image(128));

    expect(Object.keys(capturedFeeds).sort()).toEqual([
      "images",
      "letterbox_pad",
      "letterbox_scale",
      "source_image",
    ]);
    expect(capturedFeeds.source_image?.dims).toEqual([1, 3, 128, 128]);
    expect([...(capturedFeeds.letterbox_scale?.data as Float32Array)]).toEqual([0.5]);
    expect([...(capturedFeeds.letterbox_pad?.data as Float32Array)]).toEqual([0, 0]);
  });

  it("forwards the padding a non-square image needed", async () => {
    serveOutputs();
    cannedLetterbox = { scale: 1, padLeft: 0, padTop: 16 };
    const pipeline = await DetectClassify.create(
      pipelineModel({ crop_source: "original" }),
    );
    await pipeline.predict(new RGBImage(new Uint8Array(64 * 32 * 3), 64, 32));

    expect([...(capturedFeeds.letterbox_scale?.data as Float32Array)]).toEqual([1]);
    expect([...(capturedFeeds.letterbox_pad?.data as Float32Array)]).toEqual([0, 16]);
  });
});

describe("DetectClassify results", () => {
  it("reads only the rows the graph reports as real", async () => {
    serveOutputs({ count: 2 });
    const pipeline = await DetectClassify.create(pipelineModel());

    expect((await pipeline.predict(image()))[0]?.length).toBe(2);
  });

  it("yields an empty envelope when nothing was detected", async () => {
    serveOutputs({ count: 0 });
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.predict(image()))[0];

    expect(result?.length).toBe(0);
    expect(result?.boxes.length).toBe(0);
  });

  it("clamps a count that overruns the rows the graph returned", async () => {
    serveOutputs({ count: 99 });
    const pipeline = await DetectClassify.create(pipelineModel());

    expect((await pipeline.predict(image()))[0]?.length).toBe(4);
  });

  it("maps boxes out of letterbox space", async () => {
    serveOutputs();
    cannedLetterbox = { scale: 0.5, padLeft: 0, padTop: 0 };
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.predict(image(128)))[0];

    expect(result?.get(0)?.box.asIntXyxy()).toEqual([16, 16, 48, 48]);
  });

  it("subtracts the letterbox padding before undoing the scale", async () => {
    serveOutputs({ boxes: [10, 20, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], count: 1 });
    cannedLetterbox = { scale: 0.5, padLeft: 4, padTop: 8 };
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.predict(image(128)))[0];

    expect(result?.get(0)?.box.asIntXyxy()).toEqual([12, 24, 52, 64]);
  });

  it("clips boxes to the source image", async () => {
    serveOutputs({ boxes: [-20, -20, 200, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], count: 1 });
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.predict(image()))[0];

    expect(result?.get(0)?.box.asIntXyxy()).toEqual([0, 0, 64, 64]);
  });

  it("carries the crop on each detection", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    const crop = (await pipeline.predict(image()))[0]?.get(0)?.croppedImage;

    expect([crop?.width, crop?.height]).toEqual([16, 16]);
  });

  it("exposes the bulk box view", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.predict(image()))[0];

    expect(result?.boxes.shape).toEqual([2, 4]);
    expect([...(result?.boxes.cls ?? [])]).toEqual([0, 1]);
  });

  it("times every stage", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.predict(image()))[0];

    expect(Object.keys(result?.speed ?? {}).sort()).toEqual([
      "inference",
      "load",
      "postprocess",
      "preprocess",
    ]);
  });
});

describe("DetectClassify classification", () => {
  it("applies softmax when the spec says to", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    const classification = (await pipeline.predict(image()))[0]?.get(0)?.classification;

    expect(classification?.name).toBe("red");
    expect(classification?.conf).toBeCloseTo(0.9094, 3);
  });

  it("skips softmax for a graph that already normalizes", async () => {
    serveOutputs({ probs: [0.7, 0.2, 0.1, 0.7, 0.2, 0.1, 0, 0, 0, 0, 0, 0] });
    const pipeline = await DetectClassify.create(pipelineModel({ apply_softmax: "0" }));
    const classification = (await pipeline.predict(image()))[0]?.get(0)?.classification;

    expect(classification?.conf).toBeCloseTo(0.7, 5);
  });

  it("gives each detection its own row", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.predict(image()))[0];

    expect([...(result ?? [])].map((d) => d.classification?.name)).toEqual(["red", "green"]);
  });

  it("truncates the probability list to topK", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.predict(image(), { topK: 1 }))[0];

    expect(result?.get(0)?.classification?.probabilities).toHaveLength(1);
  });

  it("names a class the recorded label map does not cover", async () => {
    serveOutputs({ probs: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0] });
    const pipeline = await DetectClassify.create(
      pipelineModel({ classifier_names: "{0: 'red'}" }),
    );
    const classification = (await pipeline.predict(image()))[0]?.get(0)?.classification;

    expect(classification?.name).toBe("class_2");
  });

  it("carries the crop it describes", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    const detection = (await pipeline.predict(image()))[0]?.get(0);

    expect(detection?.classification?.image.width).toBe(detection?.croppedImage.width);
  });
});

describe("DetectClassify filtering", () => {
  it("drops detections below an extra confidence floor", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.predict(image(), { confThreshold: 0.5 }))[0];

    expect([...(result ?? [])].map((d) => d.name)).toEqual(["cat"]);
  });

  it("keeps only the allowed detector classes", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.predict(image(), { classes: [1] }))[0];

    expect([...(result ?? [])].map((d) => d.name)).toEqual(["dog"]);
  });

  it("keeps nothing for an empty allowlist", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());

    expect((await pipeline.predict(image(), { classes: [] }))[0]?.length).toBe(0);
  });

  it("call() forwards to predict()", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel());
    const result = (await pipeline.call(image(), { classes: [1] }))[0];

    expect([...(result ?? [])].map((d) => d.name)).toEqual(["dog"]);
  });
});

describe("DetectClassify raiseOnEmpty", () => {
  it("returns an empty envelope by default", async () => {
    serveOutputs({ count: 0 });
    const pipeline = await DetectClassify.create(pipelineModel());

    expect((await pipeline.predict(image()))[0]?.length).toBe(0);
  });

  it("throws when the constructor asked for it", async () => {
    serveOutputs({ count: 0 });
    const pipeline = await DetectClassify.create(pipelineModel(), { raiseOnEmpty: true });

    await expect(pipeline.predict(image())).rejects.toThrow(NoDetectionsError);
  });

  it("stays quiet when something was detected", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel(), { raiseOnEmpty: true });

    expect((await pipeline.predict(image()))[0]?.length).toBe(2);
  });

  it("per-call override turns it on", async () => {
    serveOutputs({ count: 0 });
    const pipeline = await DetectClassify.create(pipelineModel());

    await expect(pipeline.predict(image(), { raiseOnEmpty: true })).rejects.toThrow(
      NoDetectionsError,
    );
  });

  it("per-call override turns it off", async () => {
    serveOutputs({ count: 0 });
    const pipeline = await DetectClassify.create(pipelineModel(), { raiseOnEmpty: true });

    expect((await pipeline.predict(image(), { raiseOnEmpty: false }))[0]?.length).toBe(0);
  });

  it("throws when a stricter per-call threshold empties the result", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel(), { raiseOnEmpty: true });

    await expect(pipeline.predict(image(), { confThreshold: 0.99 })).rejects.toThrow(
      /confThreshold=0.99/,
    );
  });

  it("reports the graph's own threshold when no override was given", async () => {
    serveOutputs({ count: 0 });
    const pipeline = await DetectClassify.create(pipelineModel({ conf_threshold: "0.4" }), {
      raiseOnEmpty: true,
    });

    await expect(pipeline.predict(image())).rejects.toThrow(/confThreshold=0.4/);
  });

  it("throws when a class filter empties the result", async () => {
    serveOutputs();
    const pipeline = await DetectClassify.create(pipelineModel(), { raiseOnEmpty: true });

    await expect(pipeline.predict(image(), { classes: [7] })).rejects.toThrow(
      /among classes \[7\]/,
    );
  });
});

describe("DetectClassify contract violations", () => {
  it("names the output a graph is missing rather than failing obscurely", async () => {
    serveOutputs();
    delete cannedOutputs.probs;
    const pipeline = await DetectClassify.create(pipelineModel());

    await expect(pipeline.predict(image())).rejects.toThrow(/missing its 'probs' output/);
  });
});
