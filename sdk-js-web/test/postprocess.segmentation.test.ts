import { describe, expect, it } from "vitest";

import { decodeYoloSeg } from "../src/postprocess/segmentation.js";
import { Mask } from "../src/types.js";

interface SegSetup {
  perAnchor: Float32Array;
  perAnchorDims: readonly number[];
  prototypes: Float32Array;
  prototypeDims: readonly number[];
  numClasses: number;
}

function buildSegOutputs({
  numClasses = 4,
  numMaskCoefs = 32,
  numAnchors = 3,
  maskH = 16,
  maskW = 16,
} = {}): SegSetup {
  const channels = 4 + numClasses + numMaskCoefs;
  const perAnchor = new Float32Array(channels * numAnchors);
  const prototypes = new Float32Array(numMaskCoefs * maskH * maskW);

  // Prototype 0: top half +5, bottom half -5 → sigmoid sharply ~1 / ~0.
  for (let y = 0; y < maskH; y++) {
    for (let x = 0; x < maskW; x++) {
      prototypes[0 * maskH * maskW + y * maskW + x] = y < maskH / 2 ? 5 : -5;
    }
  }

  return {
    perAnchor,
    perAnchorDims: [1, channels, numAnchors],
    prototypes,
    prototypeDims: [1, numMaskCoefs, maskH, maskW],
    numClasses,
  };
}

function setPerAnchor(
  data: Float32Array,
  numAnchors: number,
  channel: number,
  anchor: number,
  value: number,
): void {
  data[channel * numAnchors + anchor] = value;
}

describe("decodeYoloSeg", () => {
  it("decodes a full-image instance with prototype-shaped mask", () => {
    const setup = buildSegOutputs();
    const N = 3;
    const set = (c: number, a: number, v: number): void =>
      setPerAnchor(setup.perAnchor, N, c, a, v);

    set(0, 0, 32); set(1, 0, 32); set(2, 0, 64); set(3, 0, 64);
    set(4 + 2, 0, 0.9);
    set(4 + setup.numClasses + 0, 0, 1); // mask coef 0 = 1

    const decoded = decodeYoloSeg(
      setup.perAnchor, setup.perAnchorDims,
      setup.prototypes, setup.prototypeDims,
      {
        numClasses: setup.numClasses,
        inputWidth: 64, inputHeight: 64,
        originalWidth: 64, originalHeight: 64,
        padLeft: 0, padTop: 0, scale: 1,
        confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
      },
    );

    expect(decoded.length).toBe(1);
    const inst = decoded[0]!;
    expect(inst.classId).toBe(2);
    expect(inst.confidence).toBeCloseTo(0.9);
    expect(inst.mask).toBeInstanceOf(Mask);
    expect(inst.mask.width).toBe(64);
    expect(inst.mask.height).toBe(64);

    // Top half all 255, bottom half all 0.
    let topWhite = 0;
    let bottomBlack = 0;
    for (let y = 0; y < 32; y++) for (let x = 0; x < 64; x++)
      if (inst.mask.data[y * 64 + x] === 255) topWhite++;
    for (let y = 32; y < 64; y++) for (let x = 0; x < 64; x++)
      if (inst.mask.data[y * 64 + x] === 0) bottomBlack++;
    expect(topWhite / 2048).toBeGreaterThan(0.95);
    expect(bottomBlack / 2048).toBeGreaterThan(0.95);

    // Binary mask: only 0 and 255.
    const unique = new Set(inst.mask.data);
    for (const v of unique) {
      expect([0, 255]).toContain(v);
    }
  });

  it("keeps overlapping instances of distinct classes", () => {
    const setup = buildSegOutputs();
    const N = 3;
    const set = (c: number, a: number, v: number): void =>
      setPerAnchor(setup.perAnchor, N, c, a, v);

    set(0, 0, 32); set(1, 0, 32); set(2, 0, 64); set(3, 0, 64);
    set(4 + 2, 0, 0.9);
    set(4 + setup.numClasses + 0, 0, 1);

    set(0, 1, 32); set(1, 1, 32); set(2, 1, 64); set(3, 1, 64);
    set(4 + 1, 1, 0.85);
    set(4 + setup.numClasses + 0, 1, 1);

    const decoded = decodeYoloSeg(
      setup.perAnchor, setup.perAnchorDims,
      setup.prototypes, setup.prototypeDims,
      {
        numClasses: setup.numClasses,
        inputWidth: 64, inputHeight: 64,
        originalWidth: 64, originalHeight: 64,
        padLeft: 0, padTop: 0, scale: 1,
        confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
      },
    );
    const cls = decoded.map((d) => d.classId).sort();
    expect(cls).toEqual([1, 2]);
  });

  it("returns [] when nothing passes confThreshold", () => {
    const setup = buildSegOutputs();
    setPerAnchor(setup.perAnchor, 3, 4 + 0, 0, 0.05);
    const decoded = decodeYoloSeg(
      setup.perAnchor, setup.perAnchorDims,
      setup.prototypes, setup.prototypeDims,
      {
        numClasses: setup.numClasses,
        inputWidth: 64, inputHeight: 64,
        originalWidth: 64, originalHeight: 64,
        padLeft: 0, padTop: 0, scale: 1,
        confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
      },
    );
    expect(decoded).toEqual([]);
  });

  it("rejects mismatched numClasses + numMaskCoefs", () => {
    const setup = buildSegOutputs({ numClasses: 4, numMaskCoefs: 32 });
    expect(() =>
      decodeYoloSeg(
        setup.perAnchor, setup.perAnchorDims,
        setup.prototypes, setup.prototypeDims,
        {
          numClasses: 10, // wrong
          inputWidth: 64, inputHeight: 64,
          originalWidth: 64, originalHeight: 64,
          padLeft: 0, padTop: 0, scale: 1,
          confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
        },
      ),
    ).toThrow(/does not match/);
  });

  it("respects maskThreshold for binarization", () => {
    // All-zero perAnchor + all-zero prototypes → matmul = 0 → sigmoid = 0.5.
    const numClasses = 4;
    const numMaskCoefs = 32;
    const numAnchors = 1;
    const maskH = 16;
    const maskW = 16;
    const channels = 4 + numClasses + numMaskCoefs;
    const perAnchor = new Float32Array(channels * numAnchors);
    const prototypes = new Float32Array(numMaskCoefs * maskH * maskW);
    perAnchor[0 * numAnchors] = 32;
    perAnchor[1 * numAnchors] = 32;
    perAnchor[2 * numAnchors] = 64;
    perAnchor[3 * numAnchors] = 64;
    perAnchor[(4 + 0) * numAnchors] = 0.9;

    const dimsPA = [1, channels, numAnchors];
    const dimsP = [1, numMaskCoefs, maskH, maskW];

    const low = decodeYoloSeg(perAnchor, dimsPA, prototypes, dimsP, {
      numClasses, inputWidth: 64, inputHeight: 64,
      originalWidth: 64, originalHeight: 64,
      padLeft: 0, padTop: 0, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
      maskThreshold: 0.4,
    });
    const high = decodeYoloSeg(perAnchor, dimsPA, prototypes, dimsP, {
      numClasses, inputWidth: 64, inputHeight: 64,
      originalWidth: 64, originalHeight: 64,
      padLeft: 0, padTop: 0, scale: 1,
      confThreshold: 0.25, iouThreshold: 0.45, maxDetections: 10,
      maskThreshold: 0.6,
    });

    expect(low[0]!.mask.data.every((v) => v === 255)).toBe(true);
    expect(high[0]!.mask.data.every((v) => v === 0)).toBe(true);
  });
});
