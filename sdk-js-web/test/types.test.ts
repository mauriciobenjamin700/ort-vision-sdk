import { describe, expect, it } from "vitest";

import { ImageLoadError } from "../src/core/exceptions.js";
import { BoundingBox, Mask, RGBImage } from "../src/types.js";

describe("BoundingBox", () => {
  it("computes width/height/area for normal boxes", () => {
    const b = new BoundingBox(10, 20, 60, 80);
    expect(b.width).toBe(50);
    expect(b.height).toBe(60);
    expect(b.area).toBe(3000);
  });

  it("clamps negative dimensions to zero", () => {
    const b = new BoundingBox(60, 80, 10, 20);
    expect(b.width).toBe(0);
    expect(b.height).toBe(0);
    expect(b.area).toBe(0);
  });

  it("formats as xyxy / xywh / int xyxy", () => {
    const b = new BoundingBox(1.5, 2.5, 11.7, 22.9);
    expect(b.asXyxy()).toEqual([1.5, 2.5, 11.7, 22.9]);
    expect(b.asXywh()[0]).toBeCloseTo(1.5);
    expect(b.asXywh()[2]).toBeCloseTo(10.2);
    expect(b.asIntXyxy()).toEqual([1, 2, 11, 22]);
  });
});

describe("RGBImage", () => {
  it("accepts data of correct length", () => {
    const img = new RGBImage(new Uint8Array(2 * 3 * 3), 2, 3);
    expect(img.width).toBe(2);
    expect(img.height).toBe(3);
    expect(img.data.length).toBe(18);
  });

  it("rejects mismatched data length", () => {
    expect(() => new RGBImage(new Uint8Array(5), 2, 3)).toThrow(ImageLoadError);
  });
});

describe("Mask", () => {
  it("accepts data of correct length", () => {
    const m = new Mask(new Uint8Array(4 * 5), 4, 5);
    expect(m.width).toBe(4);
    expect(m.height).toBe(5);
    expect(m.data.length).toBe(20);
  });

  it("rejects mismatched data length", () => {
    expect(() => new Mask(new Uint8Array(3), 4, 5)).toThrow(ImageLoadError);
  });

  it("allows zero-sized masks for empty boxes", () => {
    const m = new Mask(new Uint8Array(0), 0, 0);
    expect(m.data.length).toBe(0);
  });
});
