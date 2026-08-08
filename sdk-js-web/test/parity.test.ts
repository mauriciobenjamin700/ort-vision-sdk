/**
 * Parity tests against the shared Python/Web fixtures.
 *
 * The fixtures in `fixtures/parity/` at the repository root record inputs and
 * the outputs the **Python** SDK produces. This suite feeds the same inputs to
 * the web implementation and requires the same outputs, which is what keeps the
 * two published artifacts from drifting apart — they promise the same API and
 * the same numbers, and until these fixtures existed nothing checked the second
 * half of that promise.
 *
 * Floats are compared with a tolerance: Python computes in `float32` while
 * JavaScript numbers are `float64`, so the last bits legitimately differ.
 * Integers, kept-index lists and mask bitmaps are compared exactly — those are
 * decisions, not measurements, and a difference there is a real divergence.
 *
 * Only postprocessing is covered. Preprocessing cannot be: `letterbox` resizes
 * through a canvas here and through PIL in Python, and those resamplers do not
 * agree pixel for pixel (nor is a canvas available under Node).
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { softmax, topK } from "../src/postprocess/classification.js";
import { batchedNms, decodeYolo, nms } from "../src/postprocess/detection.js";
import { decodeYoloSeg } from "../src/postprocess/segmentation.js";
import { modelNames } from "../src/core/metadata.js";

interface TensorEntry {
  readonly dims: number[];
  readonly data: number[];
}

interface Fixture {
  readonly tensors: Record<string, TensorEntry>;
  readonly nms: NmsCase[];
  readonly batchedNms: BatchedNmsCase[];
  readonly decodeYolo: DecodeYoloCase[];
  readonly decodeYoloSeg: DecodeYoloSegCase[];
  readonly softmax: SoftmaxCase[];
  readonly topk: TopkCase[];
  readonly modelNames: ModelNamesCase[];
}

interface NmsCase {
  readonly name: string;
  readonly boxes: number[];
  readonly scores: number[];
  readonly iouThreshold: number;
  readonly expectedKeep: number[];
}

interface BatchedNmsCase extends NmsCase {
  readonly classIds: number[];
}

interface ExpectedDetection {
  readonly bbox: number[];
  readonly classId: number;
  readonly confidence: number;
}

interface ExpectedInstance extends ExpectedDetection {
  readonly maskWidth: number;
  readonly maskHeight: number;
  readonly maskBits: string;
}

interface DecodeYoloCase {
  readonly name: string;
  readonly tensor: string;
  readonly options: {
    readonly originalWidth: number;
    readonly originalHeight: number;
    readonly padLeft: number;
    readonly padTop: number;
    readonly scale: number;
    readonly confThreshold: number;
    readonly iouThreshold: number;
    readonly maxDetections: number;
  };
  readonly expected: ExpectedDetection[];
}

interface DecodeYoloSegCase {
  readonly name: string;
  readonly perAnchorTensor: string;
  readonly prototypeTensor: string;
  readonly options: DecodeYoloCase["options"] & {
    readonly numClasses: number;
    readonly inputWidth: number;
    readonly inputHeight: number;
    readonly maskThreshold: number;
  };
  readonly expected: ExpectedInstance[];
}

interface SoftmaxCase {
  readonly name: string;
  readonly logits: number[];
  readonly expected: number[];
}

interface TopkCase {
  readonly name: string;
  readonly probabilities: number[];
  readonly k: number | null;
  readonly expectedIndices: number[];
  readonly expectedValues: number[];
}

interface ModelNamesCase {
  readonly name: string;
  readonly raw: string;
  readonly expected: Record<string, string> | null;
}

const FIXTURE_URL = new URL("../../fixtures/parity/postprocess.json", import.meta.url);
const fixture = JSON.parse(readFileSync(FIXTURE_URL, "utf-8")) as Fixture;

/** Relative tolerance for float comparisons — see the module docstring. */
const FLOAT_TOLERANCE = 1e-5;

/** Materialize one of the fixture's shared tensors as a `Float32Array`. */
function loadTensor(name: string): { data: Float32Array; dims: number[] } {
  const entry = fixture.tensors[name];
  if (entry === undefined) {
    throw new Error(`parity fixture has no tensor named ${name}`);
  }
  return { data: Float32Array.from(entry.data), dims: entry.dims };
}

/** Rebuild the expected binary mask from a fixture's `maskBits` string. */
function expectedMask(instance: ExpectedInstance): Uint8Array {
  const out = new Uint8Array(instance.maskBits.length);
  for (let i = 0; i < instance.maskBits.length; i++) {
    out[i] = instance.maskBits[i] === "1" ? 255 : 0;
  }
  return out;
}

function expectCloseArray(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i] as number, 5);
  }
}

describe("nms parity", () => {
  for (const testCase of fixture.nms) {
    it(testCase.name, () => {
      const keep = nms(
        Float32Array.from(testCase.boxes),
        Float32Array.from(testCase.scores),
        testCase.iouThreshold,
      );

      expect(Array.from(keep)).toEqual(testCase.expectedKeep);
    });
  }
});

describe("batchedNms parity", () => {
  for (const testCase of fixture.batchedNms) {
    it(testCase.name, () => {
      const keep = batchedNms(
        Float32Array.from(testCase.boxes),
        Float32Array.from(testCase.scores),
        Int32Array.from(testCase.classIds),
        testCase.iouThreshold,
      );

      expect(Array.from(keep)).toEqual(testCase.expectedKeep);
    });
  }
});

describe("decodeYolo parity", () => {
  for (const testCase of fixture.decodeYolo) {
    it(testCase.name, () => {
      const { data, dims } = loadTensor(testCase.tensor);

      const decoded = decodeYolo(data, dims, testCase.options);

      expect(decoded).toHaveLength(testCase.expected.length);
      decoded.forEach((actual, i) => {
        const want = testCase.expected[i] as ExpectedDetection;
        expect(actual.classId).toBe(want.classId);
        expect(actual.confidence).toBeCloseTo(want.confidence, 5);
        expectCloseArray(actual.bbox.asXyxy(), want.bbox);
      });
    });
  }
});

describe("decodeYoloSeg parity", () => {
  for (const testCase of fixture.decodeYoloSeg) {
    it(testCase.name, () => {
      const perAnchor = loadTensor(testCase.perAnchorTensor);
      const prototypes = loadTensor(testCase.prototypeTensor);

      const decoded = decodeYoloSeg(
        perAnchor.data,
        perAnchor.dims,
        prototypes.data,
        prototypes.dims,
        testCase.options,
      );

      expect(decoded).toHaveLength(testCase.expected.length);
      decoded.forEach((actual, i) => {
        const want = testCase.expected[i] as ExpectedInstance;
        expect(actual.classId).toBe(want.classId);
        expect(actual.confidence).toBeCloseTo(want.confidence, 5);
        expectCloseArray(actual.bbox.asXyxy(), want.bbox);
        expect(actual.mask.width).toBe(want.maskWidth);
        expect(actual.mask.height).toBe(want.maskHeight);
        expect(Array.from(actual.mask.data)).toEqual(Array.from(expectedMask(want)));
      });
    });
  }
});

describe("softmax parity", () => {
  for (const testCase of fixture.softmax) {
    it(testCase.name, () => {
      expectCloseArray(softmax(Float32Array.from(testCase.logits)), testCase.expected);
    });
  }
});

describe("topK parity", () => {
  for (const testCase of fixture.topk) {
    it(testCase.name, () => {
      const result = topK(Float32Array.from(testCase.probabilities), testCase.k);

      expect(Array.from(result.indices)).toEqual(testCase.expectedIndices);
      expectCloseArray(result.values, testCase.expectedValues);
    });
  }
});

describe("modelNames parity", () => {
  for (const testCase of fixture.modelNames) {
    it(testCase.name, () => {
      const parsed = modelNames(testCase.raw ? { names: testCase.raw } : {});

      if (testCase.expected === null) {
        expect(parsed).toBeNull();
      } else {
        const asStringKeys: Record<string, string> = {};
        for (const [key, value] of Object.entries(parsed ?? {})) {
          asStringKeys[key] = value;
        }
        expect(asStringKeys).toEqual(testCase.expected);
      }
    });
  }
});

describe("fixture integrity", () => {
  it("uses the tolerance only for floats", () => {
    expect(FLOAT_TOLERANCE).toBeLessThan(1e-4);
  });

  it("covers every section", () => {
    expect(fixture.nms.length).toBeGreaterThan(0);
    expect(fixture.batchedNms.length).toBeGreaterThan(0);
    expect(fixture.decodeYolo.length).toBeGreaterThan(0);
    expect(fixture.decodeYoloSeg.length).toBeGreaterThan(0);
    expect(fixture.softmax.length).toBeGreaterThan(0);
    expect(fixture.topk.length).toBeGreaterThan(0);
    expect(fixture.modelNames.length).toBeGreaterThan(0);
  });
});
