import { describe, expect, it, vi } from "vitest";

import {
  classificationNumClasses,
  declaredShapesFrom,
  detectionNumClasses,
  resolveInputSize,
  spatialInputSize,
} from "../src/core/graph.js";

/**
 * Build the ORT metadata shape the runtime reports, without loading a model.
 *
 * @param shape Declared dimensions, strings standing for symbolic axes.
 * @returns A single-input metadata array.
 */
function tensorMetadata(shape: readonly (number | string)[]): Parameters<typeof declaredShapesFrom>[0] {
  return [{ name: "images", isTensor: true, type: "float32", shape }] as never;
}

describe("declaredShapesFrom", () => {
  it("keeps static dims and nulls symbolic ones", () => {
    expect(declaredShapesFrom(tensorMetadata([1, 3, 224, 224]))).toEqual([[1, 3, 224, 224]]);
    expect(declaredShapesFrom(tensorMetadata(["batch", 3, "height", "width"]))).toEqual([
      [null, 3, null, null],
    ]);
  });

  it("treats non-positive and fractional dims as undeclared", () => {
    expect(declaredShapesFrom(tensorMetadata([-1, 3, 0, 224.5]))).toEqual([[null, 3, null, null]]);
  });

  it("yields an empty shape for non-tensor values and no metadata", () => {
    expect(declaredShapesFrom([{ name: "seq", isTensor: false }] as never)).toEqual([[]]);
    expect(declaredShapesFrom(undefined)).toEqual([]);
  });
});

describe("spatialInputSize", () => {
  it("reads width and height out of a static NCHW shape", () => {
    expect(spatialInputSize([1, 3, 224, 224])).toEqual([224, 224]);
    expect(spatialInputSize([1, 3, 480, 640])).toEqual([640, 480]);
  });

  it("returns null when the shape pins no resolution", () => {
    expect(spatialInputSize([1, 3, null, null])).toBeNull();
    expect(spatialInputSize([1, 3, 224])).toBeNull();
    expect(spatialInputSize([])).toBeNull();
  });
});

describe("resolveInputSize", () => {
  it("prefers what the graph declares over the requested size", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    expect(
      resolveInputSize({
        graphShape: [1, 3, 224, 224],
        requested: [640, 640],
        fallback: [640, 640],
      }),
    ).toEqual([224, 224]);
    expect(warn).toHaveBeenCalledOnce();

    warn.mockRestore();
  });

  it("stays quiet when the requested size already matches the graph", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    expect(
      resolveInputSize({
        graphShape: [1, 3, 224, 224],
        requested: [224, 224],
        fallback: [640, 640],
      }),
    ).toEqual([224, 224]);
    expect(warn).not.toHaveBeenCalled();

    warn.mockRestore();
  });

  it("uses the graph when nothing was requested", () => {
    expect(resolveInputSize({ graphShape: [1, 3, 320, 320], fallback: [640, 640] })).toEqual([
      320, 320,
    ]);
  });

  it("falls back to the requested size, then the default, for a dynamic graph", () => {
    expect(
      resolveInputSize({
        graphShape: [1, 3, null, null],
        requested: [512, 512],
        fallback: [640, 640],
      }),
    ).toEqual([512, 512]);
    expect(resolveInputSize({ graphShape: [1, 3, null, null], fallback: [640, 640] })).toEqual([
      640, 640,
    ]);
    expect(resolveInputSize({ fallback: [640, 640] })).toEqual([640, 640]);
  });
});

describe("detectionNumClasses", () => {
  it("reads nc off a YOLO head declaring (B, 4 + nc, N)", () => {
    expect(detectionNumClasses([1, 84, 8400])).toBe(80);
    expect(detectionNumClasses([1, 5, 8400])).toBe(1);
    expect(detectionNumClasses([1, 6, 2100])).toBe(2);
  });

  it("returns null when the shape cannot pin it", () => {
    expect(detectionNumClasses([1, null, null])).toBeNull();
    expect(detectionNumClasses([])).toBeNull();
    // 4 channels would leave zero classes after the box coordinates.
    expect(detectionNumClasses([1, 4, 8400])).toBeNull();
  });
});

describe("classificationNumClasses", () => {
  it("reads nc off a classifier head declaring (B, nc)", () => {
    expect(classificationNumClasses([1, 2])).toBe(2);
    expect(classificationNumClasses([1, 1000])).toBe(1000);
  });

  it("returns null when the last axis is dynamic or absent", () => {
    expect(classificationNumClasses([1, null])).toBeNull();
    expect(classificationNumClasses([])).toBeNull();
  });
});
