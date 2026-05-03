import { describe, expect, it } from "vitest";

import {
  batchedNms,
  decodeYolo,
  decodeYoloAnchors,
  nms,
} from "../src/postprocess/detection.js";
import { BoundingBox } from "../src/types.js";

interface AnchorSpec {
  cx: number;
  cy: number;
  w: number;
  h: number;
  classId: number;
  score: number;
}

function buildYoloOutput(
  anchors: AnchorSpec[],
  numClasses: number,
  extraChannels = 0,
): { data: Float32Array; dims: readonly number[] } {
  const n = anchors.length;
  const channels = 4 + numClasses + extraChannels;
  const data = new Float32Array(channels * n);
  const set = (c: number, a: number, v: number): void => {
    data[c * n + a] = v;
  };
  for (let a = 0; a < n; a++) {
    const spec = anchors[a]!;
    set(0, a, spec.cx);
    set(1, a, spec.cy);
    set(2, a, spec.w);
    set(3, a, spec.h);
    set(4 + spec.classId, a, spec.score);
  }
  return { data, dims: [1, channels, n] };
}

describe("nms", () => {
  it("returns empty for empty input", () => {
    const keep = nms(new Float32Array(0), new Float32Array(0), 0.5);
    expect(keep.length).toBe(0);
  });

  it("suppresses overlapping lower-score box", () => {
    const boxes = new Float32Array([0, 0, 10, 10, 1, 1, 9, 9]);
    const scores = new Float32Array([0.9, 0.8]);
    const keep = nms(boxes, scores, 0.5);
    expect(Array.from(keep)).toEqual([0]);
  });

  it("keeps distant boxes", () => {
    const boxes = new Float32Array([0, 0, 10, 10, 100, 100, 110, 110]);
    const scores = new Float32Array([0.9, 0.7]);
    const keep = nms(boxes, scores, 0.5);
    expect(Array.from(keep)).toEqual([0, 1]);
  });

  it("returns descending score order", () => {
    const boxes = new Float32Array([
      0, 0, 5, 5,
      10, 0, 15, 5,
      20, 0, 25, 5,
    ]);
    const scores = new Float32Array([0.3, 0.9, 0.6]);
    const keep = nms(boxes, scores, 0.5);
    expect(Array.from(keep)).toEqual([1, 2, 0]);
  });
});

describe("batchedNms", () => {
  it("does not suppress across classes", () => {
    const boxes = new Float32Array([0, 0, 10, 10, 1, 1, 9, 9]);
    const scores = new Float32Array([0.9, 0.85]);
    const idxs = new Int32Array([0, 1]);
    const keep = batchedNms(boxes, scores, idxs, 0.5);
    expect(Array.from(keep).sort()).toEqual([0, 1]);
  });

  it("suppresses within the same class", () => {
    const boxes = new Float32Array([0, 0, 10, 10, 1, 1, 9, 9]);
    const scores = new Float32Array([0.9, 0.85]);
    const idxs = new Int32Array([0, 0]);
    const keep = batchedNms(boxes, scores, idxs, 0.5);
    expect(Array.from(keep)).toEqual([0]);
  });

  it("returns empty for empty input", () => {
    const keep = batchedNms(
      new Float32Array(0),
      new Float32Array(0),
      new Int32Array(0),
      0.5,
    );
    expect(keep.length).toBe(0);
  });
});

describe("decodeYoloAnchors", () => {
  it("filters anchors below confThreshold", () => {
    const { data, dims } = buildYoloOutput(
      [
        { cx: 320, cy: 320, w: 100, h: 100, classId: 2, score: 0.9 },
        { cx: 50, cy: 50, w: 40, h: 40, classId: 1, score: 0.05 },
      ],
      4,
    );
    const r = decodeYoloAnchors(data, dims, {
      numClasses: 4,
      originalWidth: 640, originalHeight: 480,
      padLeft: 0, padTop: 80, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
    });
    expect(Array.from(r.anchorIndices)).toEqual([0]);
    expect(Array.from(r.classIds)).toEqual([2]);
    expect(r.confidences[0]).toBeCloseTo(0.9);
  });

  it("keeps overlapping boxes of different classes (per-class NMS)", () => {
    const { data, dims } = buildYoloOutput(
      [
        { cx: 320, cy: 320, w: 100, h: 100, classId: 2, score: 0.9 },
        { cx: 320, cy: 320, w: 100, h: 100, classId: 1, score: 0.85 },
      ],
      4,
    );
    const r = decodeYoloAnchors(data, dims, {
      numClasses: 4,
      originalWidth: 640, originalHeight: 480,
      padLeft: 0, padTop: 80, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
    });
    expect(Array.from(r.classIds).sort()).toEqual([1, 2]);
  });

  it("orders survivors by descending confidence", () => {
    const { data, dims } = buildYoloOutput(
      [
        { cx: 10, cy: 10, w: 5, h: 5, classId: 0, score: 0.4 },
        { cx: 200, cy: 200, w: 5, h: 5, classId: 1, score: 0.9 },
        { cx: 400, cy: 400, w: 5, h: 5, classId: 2, score: 0.6 },
      ],
      4,
    );
    const r = decodeYoloAnchors(data, dims, {
      numClasses: 4,
      originalWidth: 640, originalHeight: 480,
      padLeft: 0, padTop: 0, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
    });
    const confs = Array.from(r.confidences);
    expect(confs).toEqual([...confs].sort((a, b) => b - a));
  });

  it("caps at maxDetections", () => {
    const { data, dims } = buildYoloOutput(
      [
        { cx: 10, cy: 10, w: 5, h: 5, classId: 0, score: 0.9 },
        { cx: 200, cy: 200, w: 5, h: 5, classId: 1, score: 0.85 },
        { cx: 400, cy: 400, w: 5, h: 5, classId: 2, score: 0.8 },
      ],
      4,
    );
    const r = decodeYoloAnchors(data, dims, {
      numClasses: 4,
      originalWidth: 640, originalHeight: 480,
      padLeft: 0, padTop: 0, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 2,
    });
    expect(r.anchorIndices.length).toBe(2);
  });
});

describe("decodeYolo", () => {
  it("returns BoundingBox + class + conf with letterbox-corrected coords", () => {
    const { data, dims } = buildYoloOutput(
      [
        { cx: 320, cy: 320, w: 100, h: 100, classId: 2, score: 0.9 },
        { cx: 330, cy: 330, w: 100, h: 100, classId: 2, score: 0.8 }, // NMS-suppressed
        { cx: 50, cy: 200, w: 40, h: 40, classId: 1, score: 0.85 },
      ],
      4,
    );
    const dets = decodeYolo(data, dims, {
      originalWidth: 640, originalHeight: 480,
      padLeft: 0, padTop: 80, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
    });
    expect(dets.length).toBe(2);
    expect(dets[0]!.bbox).toBeInstanceOf(BoundingBox);
    expect(dets[0]!.classId).toBe(2);
    expect(dets[0]!.confidence).toBeCloseTo(0.9);
    expect(dets[0]!.bbox.asXyxy().map((v) => Math.round(v))).toEqual([270, 190, 370, 290]);
  });

  it("returns [] when nothing passes", () => {
    const { data, dims } = buildYoloOutput(
      [{ cx: 10, cy: 10, w: 5, h: 5, classId: 0, score: 0.05 }],
      4,
    );
    const dets = decodeYolo(data, dims, {
      originalWidth: 640, originalHeight: 480,
      padLeft: 0, padTop: 0, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
    });
    expect(dets).toEqual([]);
  });

  it("rejects invalid channel count", () => {
    // 4 channels = boxes only, no class scores.
    expect(() =>
      decodeYolo(new Float32Array(4 * 3), [1, 4, 3], {
        originalWidth: 100, originalHeight: 100,
        padLeft: 0, padTop: 0, scale: 1,
        confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
      }),
    ).toThrow(/channel count/);
  });
});
