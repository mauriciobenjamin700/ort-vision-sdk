import { describe, expect, it } from "vitest";

import {
  decodeYolo,
  decodeYoloAnchors,
  decodeYoloV8,
  decodeYoloV8Anchors,
} from "../src/postprocess/detection.js";
import {
  decodeYoloSeg,
  decodeYoloV8Seg,
} from "../src/postprocess/segmentation.js";

function detectionOutput(): { data: Float32Array; dims: readonly number[] } {
  // 1 anchor, 4 classes — channel layout (4 + 4) * 1.
  const numClasses = 4;
  const n = 1;
  const channels = 4 + numClasses;
  const data = new Float32Array(channels * n);
  data[0] = 320; data[n] = 320; data[2 * n] = 100; data[3 * n] = 100;
  data[(4 + 2) * n] = 0.9;
  return { data, dims: [1, channels, n] };
}

describe("decodeYoloV8 (deprecated alias)", () => {
  it("matches decodeYolo's output", () => {
    const { data, dims } = detectionOutput();
    const opts = {
      originalWidth: 640, originalHeight: 480,
      padLeft: 0, padTop: 0, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
    };
    const oldR = decodeYoloV8(data, dims, opts);
    const newR = decodeYolo(data, dims, opts);
    expect(oldR.length).toBe(newR.length);
    expect(oldR[0]!.bbox.asXyxy()).toEqual(newR[0]!.bbox.asXyxy());
    expect(oldR[0]!.classId).toBe(newR[0]!.classId);
    expect(oldR[0]!.confidence).toBe(newR[0]!.confidence);
  });
});

describe("decodeYoloV8Anchors (deprecated alias)", () => {
  it("matches decodeYoloAnchors's output", () => {
    const { data, dims } = detectionOutput();
    const opts = {
      numClasses: 4,
      originalWidth: 640, originalHeight: 480,
      padLeft: 0, padTop: 0, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
    };
    const oldR = decodeYoloV8Anchors(data, dims, opts);
    const newR = decodeYoloAnchors(data, dims, opts);
    expect(Array.from(oldR.anchorIndices)).toEqual(Array.from(newR.anchorIndices));
    expect(Array.from(oldR.classIds)).toEqual(Array.from(newR.classIds));
    expect(Array.from(oldR.confidences)).toEqual(Array.from(newR.confidences));
  });
});

describe("decodeYoloV8Seg (deprecated alias)", () => {
  it("matches decodeYoloSeg's output", () => {
    const numClasses = 4;
    const numMaskCoefs = 32;
    const n = 1;
    const maskH = 16;
    const maskW = 16;
    const channels = 4 + numClasses + numMaskCoefs;

    const perAnchor = new Float32Array(channels * n);
    const prototypes = new Float32Array(numMaskCoefs * maskH * maskW);
    for (let y = 0; y < maskH; y++) for (let x = 0; x < maskW; x++) {
      prototypes[0 * maskH * maskW + y * maskW + x] = y < maskH / 2 ? 5 : -5;
    }
    perAnchor[0] = 32;
    perAnchor[n] = 32;
    perAnchor[2 * n] = 64;
    perAnchor[3 * n] = 64;
    perAnchor[(4 + 2) * n] = 0.9;
    perAnchor[(4 + numClasses + 0) * n] = 1;

    const opts = {
      numClasses,
      inputWidth: 64, inputHeight: 64,
      originalWidth: 64, originalHeight: 64,
      padLeft: 0, padTop: 0, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
    };
    const oldR = decodeYoloV8Seg(
      perAnchor, [1, channels, n], prototypes, [1, numMaskCoefs, maskH, maskW], opts,
    );
    const newR = decodeYoloSeg(
      perAnchor, [1, channels, n], prototypes, [1, numMaskCoefs, maskH, maskW], opts,
    );
    expect(oldR.length).toBe(newR.length);
    expect(oldR[0]!.classId).toBe(newR[0]!.classId);
    expect(oldR[0]!.confidence).toBe(newR[0]!.confidence);
    expect(Array.from(oldR[0]!.mask.data)).toEqual(Array.from(newR[0]!.mask.data));
  });
});
