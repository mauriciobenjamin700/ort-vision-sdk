import { describe, expect, it } from "vitest";

import {
  Boxes,
  ClassificationResults,
  DetectionResults,
  Masks,
  Probs,
  SegmentationResults,
} from "../src/results.js";
import { BoundingBox, Mask, RGBImage } from "../src/types.js";
import type {
  ClassificationResult,
  DetectionResult,
  SegmentationResult,
} from "../src/types.js";

const makeRGB = (w = 4, h = 4): RGBImage =>
  new RGBImage(new Uint8Array(w * h * 3), w, h);

describe("Boxes", () => {
  it("derives xywh from xyxy", () => {
    const xyxy = new Float32Array([10, 20, 60, 80, 0, 0, 100, 50]);
    const cls = new Int32Array([1, 2]);
    const conf = new Float32Array([0.9, 0.7]);
    const b = new Boxes(xyxy, cls, conf, [120, 200]);

    expect(b.length).toBe(2);
    expect(b.shape).toEqual([2, 4]);

    const xywh = Array.from(b.xywh);
    expect(xywh.slice(0, 4)).toEqual([35, 50, 50, 60]);
    expect(xywh.slice(4, 8)).toEqual([50, 25, 100, 50]);
  });

  it("normalizes coords to [0, 1] via origShape (h, w)", () => {
    const xyxy = new Float32Array([50, 30, 150, 90]);
    const b = new Boxes(
      xyxy,
      new Int32Array([0]),
      new Float32Array([0.8]),
      [120, 200], // (h, w)
    );
    const xyxyn = Array.from(b.xyxyn);
    expect(xyxyn).toEqual([0.25, 0.25, 0.75, 0.75]);
  });

  it("packs data as [N, 6] = [x1,y1,x2,y2,conf,cls]", () => {
    const b = new Boxes(
      new Float32Array([1, 2, 3, 4]),
      new Int32Array([7]),
      new Float32Array([0.5]),
      [10, 10],
    );
    expect(Array.from(b.data)).toEqual([1, 2, 3, 4, 0.5, 7]);
  });

  it("handles empty input", () => {
    const b = new Boxes(
      new Float32Array(0),
      new Int32Array(0),
      new Float32Array(0),
      [100, 100],
    );
    expect(b.length).toBe(0);
    expect(b.xywh.length).toBe(0);
    expect(b.xyxyn.length).toBe(0);
  });
});

/**
 * The full-sort selection `Probs` used before partial selection replaced it.
 *
 * Kept here as the oracle: the fast path has to agree with it on every input,
 * ties included, or the optimisation changed an answer instead of the cost.
 * It is also what the Python SDK still does (`np.argsort(-data, kind="stable")`),
 * so agreement here is agreement across the two SDKs.
 *
 * @param data - Per-class probabilities.
 * @param k - How many classes to select.
 * @returns Indices and values, descending by value.
 */
function fullSortTopK(data: Float32Array, k: number): { indices: number[]; values: number[] } {
  const n = Math.min(k, data.length);
  const order: number[] = [];
  for (let i = 0; i < data.length; i++) order.push(i);
  order.sort((a, b) => (data[b] as number) - (data[a] as number));
  return {
    indices: order.slice(0, n),
    values: order.slice(0, n).map((index) => data[index] as number),
  };
}

describe("Probs", () => {
  it("computes top1 / top1conf", () => {
    const p = new Probs(new Float32Array([0.1, 0.6, 0.2, 0.1]));
    expect(p.top1).toBe(1);
    expect(p.top1conf).toBeCloseTo(0.6);
  });

  it("computes top5 / top5conf descending", () => {
    const p = new Probs(new Float32Array([0.05, 0.10, 0.50, 0.25, 0.05, 0.05]));
    expect(Array.from(p.top5)).toEqual([2, 3, 1, 0, 4]);
    expect(p.top5conf[0]).toBeCloseTo(0.50);
    expect(p.top5conf[1]).toBeCloseTo(0.25);
  });

  it("truncates top5 when fewer than 5 classes", () => {
    const p = new Probs(new Float32Array([0.7, 0.3]));
    expect(Array.from(p.top5)).toEqual([0, 1]);
    expect(p.top5conf.length).toBe(2);
  });

  it("handles empty probs", () => {
    const p = new Probs(new Float32Array(0));
    expect(p.top1).toBe(0);
    expect(p.top1conf).toBe(0);
    expect(p.top5.length).toBe(0);
  });

  it("keeps the lower class index first on a tie", () => {
    const p = new Probs(Float32Array.from([0.5, 0.5, 0.5, 0.1]));
    expect(Array.from(p.top5)).toEqual([0, 1, 2, 3]);
    expect(p.top1).toBe(0);
  });

  it("agrees with a full sort across random vectors", () => {
    for (let round = 0; round < 40; round += 1) {
      const data = Float32Array.from({ length: 200 }, () => Math.round(Math.random() * 20));
      const p = new Probs(data);
      const oracle = fullSortTopK(data, 5);
      expect(Array.from(p.top5)).toEqual(oracle.indices);
      expect(Array.from(p.top5conf)).toEqual(oracle.values);
    }
  });

  it("computes each selection once and hands back the same arrays", () => {
    const p = new Probs(Float32Array.from([0.1, 0.9, 0.5]));
    expect(p.top5).toBe(p.top5);
    expect(p.top5conf).toBe(p.top5conf);
  });
});

describe("Masks", () => {
  it("iterates over per-instance masks", () => {
    const m1 = new Mask(new Uint8Array(16).fill(255), 4, 4);
    const m2 = new Mask(new Uint8Array(36), 6, 6);
    const masks = new Masks(
      [m1, m2],
      new Float32Array([0, 0, 4, 4, 10, 10, 16, 16]),
      [20, 20],
    );
    expect(masks.length).toBe(2);
    expect(masks.shape).toEqual([2]);
    const sizes = Array.from(masks).map((m) => [m.width, m.height]);
    expect(sizes).toEqual([[4, 4], [6, 6]]);
  });
});

describe("DetectionResults", () => {
  it("iterates and indexes per-instance detections", () => {
    const det: DetectionResult = {
      classId: 1,
      className: "cat",
      confidence: 0.9,
      bbox: new BoundingBox(0, 0, 10, 10),
      croppedImage: makeRGB(10, 10),
    };
    const results = new DetectionResults(
      new Boxes(
        new Float32Array([0, 0, 10, 10]),
        new Int32Array([1]),
        new Float32Array([0.9]),
        [20, 20],
      ),
      [det],
      { 1: "cat" },
      makeRGB(20, 20),
      [20, 20],
    );
    expect(results.length).toBe(1);
    expect(results.get(0)).toBe(det);
    expect([...results]).toEqual([det]);
  });
});

describe("ClassificationResults", () => {
  it("exposes Ultralytics-style cls / conf / name aliases", () => {
    const probs = new Probs(new Float32Array([0.1, 0.7, 0.2]));
    const legacy: ClassificationResult = {
      classId: 1,
      className: "dog",
      confidence: 0.7,
      image: makeRGB(),
      probabilities: [
        { classId: 1, className: "dog", probability: 0.7 },
        { classId: 2, className: "cat", probability: 0.2 },
      ],
    };
    const results = new ClassificationResults(
      probs,
      legacy,
      { 0: "fish", 1: "dog", 2: "cat" },
      makeRGB(),
      [4, 4],
    );
    expect(results.cls).toBe(1);
    expect(results.conf).toBeCloseTo(0.7);
    expect(results.name).toBe("dog");
  });

  it("falls back to class_<id> for unknown class names", () => {
    const probs = new Probs(new Float32Array([0, 1]));
    const legacy: ClassificationResult = {
      classId: 1,
      className: "?",
      confidence: 1,
      image: makeRGB(1, 1),
      probabilities: [],
    };
    const results = new ClassificationResults(
      probs, legacy, {}, makeRGB(1, 1), [1, 1],
    );
    expect(results.name).toBe("class_1");
  });
});

describe("SegmentationResults", () => {
  it("iterates and indexes per-instance segmentations", () => {
    const seg: SegmentationResult = {
      classId: 0,
      className: "person",
      confidence: 0.95,
      bbox: new BoundingBox(0, 0, 10, 10),
      mask: new Mask(new Uint8Array(100).fill(255), 10, 10),
      segmentedImage: makeRGB(10, 10),
    };
    const results = new SegmentationResults(
      new Boxes(
        new Float32Array([0, 0, 10, 10]),
        new Int32Array([0]),
        new Float32Array([0.95]),
        [20, 20],
      ),
      new Masks([seg.mask], new Float32Array([0, 0, 10, 10]), [20, 20]),
      [seg],
      { 0: "person" },
      makeRGB(20, 20),
      [20, 20],
    );
    expect(results.length).toBe(1);
    expect(results.get(0)).toBe(seg);
    expect([...results]).toEqual([seg]);
  });
});
