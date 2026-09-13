/**
 * Mapping a crop-space mask onto the box it describes.
 *
 * A `detect_segment_classify` pipeline computes masks at the resolution the
 * segmenter was exported at, which is rarely the size of the box in the
 * original image. `DetectionResult.mask` promises the contract `Segmenter`
 * already produces — a binary mask shaped to the box — so the resampling has to
 * happen before a caller ever sees it.
 *
 * The Python SDK does the same thing in `tasks/pipeline.py::_mask_to_box`, with
 * the same nearest-neighbour rule, because the two SDKs have to describe the
 * same instance identically.
 */

import { describe, expect, it } from "vitest";

import { RGBImage } from "../src/types.js";
import { maskToBox } from "../src/tasks/detectClassify.js";

/** A 4x4 crop-space mask whose top half is foreground. */
function topHalfMask(): Float32Array {
  const data = new Float32Array(16);
  data.fill(1, 0, 8);
  return data;
}

/** An RGB image of the given size, contents irrelevant to the mapping. */
function target(width: number, height: number): RGBImage {
  return new RGBImage(new Uint8Array(width * height * 3), width, height);
}

describe("maskToBox", () => {
  it("keeps the same halves when the box matches the crop", () => {
    const mask = maskToBox(topHalfMask(), 0, 4, 4, target(4, 4));

    expect(mask).not.toBeNull();
    expect(Array.from(mask!.data.subarray(0, 8))).toEqual([255, 255, 255, 255, 255, 255, 255, 255]);
    expect(Array.from(mask!.data.subarray(8))).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("upsamples to a larger box without inventing intermediate values", () => {
    const mask = maskToBox(topHalfMask(), 0, 4, 4, target(8, 8));

    expect(mask!.width).toBe(8);
    expect(mask!.height).toBe(8);
    expect(new Set(mask!.data)).toEqual(new Set([0, 255]));
    expect(Array.from(mask!.data.subarray(0, 8))).toEqual(Array(8).fill(255));
    expect(Array.from(mask!.data.subarray(56))).toEqual(Array(8).fill(0));
  });

  it("downsamples to a smaller box", () => {
    const mask = maskToBox(topHalfMask(), 0, 4, 4, target(2, 2));

    expect(mask!.width).toBe(2);
    expect(Array.from(mask!.data)).toEqual([255, 255, 0, 0]);
  });

  it("reads the row belonging to the detection asked for", () => {
    const two = new Float32Array(32);
    two.fill(1, 16, 32);

    const first = maskToBox(two, 0, 4, 4, target(4, 4));
    const second = maskToBox(two, 1, 4, 4, target(4, 4));

    expect(Array.from(first!.data)).toEqual(Array(16).fill(0));
    expect(Array.from(second!.data)).toEqual(Array(16).fill(255));
  });

  it("returns null for a box with no area", () => {
    expect(maskToBox(topHalfMask(), 0, 4, 4, target(0, 0))).toBeNull();
  });

  it("thresholds at the midpoint, so a soft value does not leak through", () => {
    const soft = new Float32Array(16).fill(0.4);

    const mask = maskToBox(soft, 0, 4, 4, target(4, 4));

    expect(Array.from(mask!.data)).toEqual(Array(16).fill(0));
  });
});
