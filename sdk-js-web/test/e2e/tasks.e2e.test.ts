/**
 * End-to-end tests against a real `onnxruntime-web` session.
 *
 * The rest of the suite mocks `onnxruntime-web`, which proves the decoding maths
 * but never that a task can load a file, feed ORT a tensor it accepts and read
 * its outputs back. These tests run the same tiny models as
 * `sdk-python/tests/test_e2e_onnx.py` and assert the same hard-coded numbers, so
 * a divergence between the two SDKs on a real model fails here.
 *
 * The values are hard-coded on purpose: if a fixture changes, both suites have
 * to be updated deliberately.
 */
import "./setup.js";

import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import {
  Classifier,
  Detector,
  hasFloat16Array,
  LabelMapError,
  OrtSession,
  RGBImage,
  Segmenter,
} from "../../src/index.js";
import { declaredShapesFrom } from "../../src/core/graph.js";
import { readModelShapes } from "../../src/core/metadata.js";
import { model, WASM } from "./models.js";

/** Expected boxes for a 64x64 input, where letterboxing is the identity. */
const SQUARE_BOXES = [
  [16, 16, 48, 48],
  [16, 16, 48, 48],
  [6, 28, 14, 36],
];

/**
 * Expected boxes for a 128x64 image letterboxed into the model's 64x64 input.
 *
 * Letterboxing scales by 0.5 and pads 16 px vertically, so undoing it maps
 * input-space `x` to `x / 0.5` and `y` to `(y - 16) / 0.5`, clipped to the
 * image. This is the case that fails if the letterbox or its inverse drifts.
 */
const WIDE_BOXES = [
  [32, 0, 96, 64],
  [32, 0, 96, 64],
  [12, 24, 28, 40],
];

/** Class ids the detector keeps after per-class NMS, in score order. */
const EXPECTED_CLS = [0, 2, 1];

/** Scores of the kept detections. */
const EXPECTED_CONF = [0.9, 0.8, 0.7];

/** Softmax of the classifier's constant logits `[1.0, 3.0, 0.5, 2.0]`. */
const SOFTMAX = [0.08536889, 0.63079554, 0.05177886, 0.23205669];

/**
 * A black RGB image.
 *
 * @param width Width in pixels.
 * @param height Height in pixels.
 * @returns The image.
 */
function black(width: number, height: number): RGBImage {
  return new RGBImage(new Uint8Array(width * height * 3), width, height);
}

/**
 * Assert two numeric sequences match element-wise within a relative tolerance.
 *
 * @param actual Values produced by the SDK.
 * @param expected Values pinned by the test.
 * @param rtol Relative tolerance.
 */
function expectClose(actual: ArrayLike<number>, expected: readonly number[], rtol = 1e-6): void {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    expect(Math.abs((actual[index] as number) - value)).toBeLessThanOrEqual(rtol * Math.abs(value) + 1e-6);
  });
}

describe("Detector against real ORT", () => {
  it("reads labels and input size from the model", async () => {
    const detector = await Detector.create(model("tiny_detector.onnx"), { providers: WASM });

    expect(detector.labels).toEqual(["cat", "dog", "bird"]);
    expect(detector.names).toEqual({ 0: "cat", 1: "dog", 2: "bird" });
    expect(detector.numClasses).toBe(3);
    expect(detector.inputSize).toEqual([64, 64]);
  });

  it("decodes the expected boxes on a square image", async () => {
    const detector = await Detector.create(model("tiny_detector.onnx"), { providers: WASM });

    const result = (await detector.predict(black(64, 64)))[0]!;

    expect(result.length).toBe(3);
    expect([...result].map((d) => d.bbox.asXyxy())).toEqual(SQUARE_BOXES);
    expect(Array.from(result.boxes.cls)).toEqual(EXPECTED_CLS);
    expectClose(result.boxes.conf, EXPECTED_CONF);
    expect([...result].map((d) => d.className)).toEqual(["cat", "bird", "dog"]);
  });

  it("undoes the letterbox geometry on a wide image", async () => {
    const detector = await Detector.create(model("tiny_detector.onnx"), { providers: WASM });

    const result = (await detector.predict(black(128, 64)))[0]!;

    expect([...result].map((d) => d.bbox.asXyxy())).toEqual(WIDE_BOXES);
    expect(Array.from(result.boxes.cls)).toEqual(EXPECTED_CLS);
    expect(result.origShape).toEqual([64, 128]);
  });

  it("crops from the original image", async () => {
    const detector = await Detector.create(model("tiny_detector.onnx"), { providers: WASM });
    const original = black(64, 64);
    for (let y = 28; y < 36; y++) {
      for (let x = 6; x < 14; x++) original.data.set([10, 20, 30], (y * 64 + x) * 3);
    }

    const smallest = (await detector.predict(original))[0]!.get(2)!;

    expect(smallest.bbox.asXyxy()).toEqual([6, 28, 14, 36]);
    expect([smallest.croppedImage.width, smallest.croppedImage.height]).toEqual([8, 8]);
    for (let i = 0; i < smallest.croppedImage.data.length; i += 3) {
      expect(Array.from(smallest.croppedImage.data.subarray(i, i + 3))).toEqual([10, 20, 30]);
    }
  });

  it("applies the confidence override and the classes filter", async () => {
    const detector = await Detector.create(model("tiny_detector.onnx"), { providers: WASM });

    const strict = (await detector.predict(black(64, 64), { confThreshold: 0.75 }))[0]!;
    const dogs = (await detector.predict(black(64, 64), { classes: [1] }))[0]!;

    expect(Array.from(strict.boxes.cls)).toEqual([0, 2]);
    expect([...dogs].map((d) => d.className)).toEqual(["dog"]);
  });

  it("runs overlapping predict() calls on one task", async () => {
    const detector = await Detector.create(model("tiny_detector.onnx"), { providers: WASM });

    const runs = await Promise.all([1, 2, 3].map(() => detector.predict(black(64, 64))));

    for (const [result] of runs) expect(Array.from(result!.boxes.cls)).toEqual(EXPECTED_CLS);
  });

  it("names the classes of a model without baked-in names", async () => {
    const detector = await Detector.create(model("tiny_detector_no_metadata.onnx"), { providers: WASM });

    expect(detector.labels).toEqual(["class_0", "class_1", "class_2"]);
  });

  it("rejects an explicit label list of the wrong length", async () => {
    await expect(
      Detector.create(model("tiny_detector_no_metadata.onnx"), { providers: WASM, labels: ["a", "b"] }),
    ).rejects.toBeInstanceOf(LabelMapError);
  });

  it("exposes the export metadata", async () => {
    const detector = await Detector.create(model("tiny_detector.onnx"), { providers: WASM });

    expect(detector.session.metadata["task"]).toBe("detect");
    expect(detector.session.metadata["imgsz"]).toBe("[64, 64]");
    expect(detector.session.metadata["names"]).toBe("{0: 'cat', 1: 'dog', 2: 'bird'}");
  });
});

describe("Classifier against real ORT", () => {
  it("reads labels and input size from the model", async () => {
    const classifier = await Classifier.create(model("tiny_classifier.onnx"), { providers: WASM });

    expect(classifier.labels).toEqual(["ant", "bee", "cow", "doe"]);
    expect(classifier.inputSize).toEqual([32, 32]);
  });

  it("returns the exact softmax probabilities", async () => {
    const classifier = await Classifier.create(model("tiny_classifier.onnx"), { providers: WASM });

    const result = (await classifier.predict(black(32, 32)))[0]!;

    expectClose(result.probs.data, SOFTMAX);
    expect(result.probs.top1).toBe(1);
    expect(result.name).toBe("bee");
  });

  it("returns the raw scores when softmax is disabled", async () => {
    const classifier = await Classifier.create(model("tiny_classifier.onnx"), {
      providers: WASM,
      applySoftmax: false,
    });

    const result = (await classifier.predict(black(32, 32)))[0]!;

    expectClose(result.probs.data, [1.0, 3.0, 0.5, 2.0]);
  });
});

describe("Segmenter against real ORT", () => {
  /**
   * The regression guard for the mask-coefficient count.
   *
   * The model's detection output is `[1, 39, 40]`: 4 box values, 3 classes and
   * 32 mask coefficients. `onnxruntime-web` >= 1.22 reports that shape, and the
   * detection formula read it as 35 classes, so the model's own 3 names were
   * rejected with `LabelMapError` and no real YOLO-seg model could be created.
   * The mocked suites could not see it: their sessions report no shapes.
   */
  it("counts classes without the mask coefficients", async () => {
    const segmenter = await Segmenter.create(model("tiny_segmenter.onnx"), { providers: WASM });

    expect(segmenter.labels).toEqual(["leaf", "stem", "root"]);
    expect(segmenter.numClasses).toBe(3);
    expect(segmenter.inputSize).toEqual([64, 64]);
  });

  it("builds the mask from the prototype geometry", async () => {
    const segmenter = await Segmenter.create(model("tiny_segmenter.onnx"), { providers: WASM });

    const result = (await segmenter.predict(black(64, 64)))[0]!;

    expect(result.length).toBe(1);
    const instance = result.get(0)!;
    expect(instance.className).toBe("stem");
    expectClose([instance.confidence], [0.9]);
    expect([instance.mask.width, instance.mask.height]).toEqual([64, 64]);
    expect(new Set(instance.mask.data)).toEqual(new Set([0, 255]));
    expect(instance.mask.data.subarray(0, 28 * 64).every((v) => v === 255)).toBe(true);
    expect(instance.mask.data.subarray(36 * 64).every((v) => v === 0)).toBe(true);
  });
});

describe("OrtSession against real ORT", () => {
  it("echoes the fed tensor through an Identity model", async () => {
    const session = await OrtSession.create(model("tiny_identity.onnx"), { providers: WASM });
    const data = Float32Array.from({ length: 48 }, (_, i) => i);

    const outputs = await session.run({ images: new ort.Tensor("float32", data, [1, 3, 4, 4]) });

    expect(session.inputName).toBe("images");
    expect(Array.from(outputs["echo"]!.data as Float32Array)).toEqual(Array.from(data));
  });
});

describe("declared shapes read from the file", () => {
  /**
   * The file reader is the fallback for runtimes below 1.22, which report no
   * shapes. Where the runtime does report them, both sources must agree, or the
   * fallback would quietly answer differently from the runtime it stands in for.
   * The peer floor is 1.22, so a runtime here that reports nothing is a failure,
   * not a case to skip.
   */
  it.each([
    "tiny_detector.onnx",
    "tiny_detector_no_metadata.onnx",
    "tiny_classifier.onnx",
    "tiny_classifier_fp16.onnx",
    "tiny_segmenter.onnx",
    "tiny_identity.onnx",
  ])("match what the runtime reports for %s", async (name) => {
    const bytes = model(name);
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    const fromFile = readModelShapes(bytes);

    expect(session.inputMetadata, "onnxruntime-web >= 1.22 reports input metadata").toBeDefined();
    expect(session.inputNames.map((n) => fromFile.inputs[n])).toEqual(
      declaredShapesFrom(session.inputMetadata),
    );
    expect(session.outputNames.map((n) => fromFile.outputs[n])).toEqual(
      declaredShapesFrom(session.outputMetadata),
    );
  });
});

describe("half-precision model against real ORT", () => {
  it.runIf(hasFloat16Array())("feeds the declared float16 and widens the outputs", async () => {
    const classifier = await Classifier.create(model("tiny_classifier_fp16.onnx"), { providers: WASM });

    await classifier.warmup(2);
    const result = (await classifier.predict(black(32, 32)))[0]!;

    expect(classifier.session.inputDtype).toBe("float16");
    expect(result.probs.data).toBeInstanceOf(Float32Array);
    expectClose(result.probs.data, SOFTMAX, 1e-3);
    expect(result.name).toBe("bee");
  });

  it.skipIf(hasFloat16Array())("refuses at create() where Float16Array is missing", async () => {
    await expect(
      Classifier.create(model("tiny_classifier_fp16.onnx"), { providers: WASM }),
    ).rejects.toThrow(/Float16Array/);
  });
});
