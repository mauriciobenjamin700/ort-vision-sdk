import { describe, expect, it } from "vitest";

import { normalize, toCHW } from "../src/preprocess/image.js";
import { ResizePipeline, resizeToTensorData, writePlanarFloat32, zeroTensorData } from "../src/preprocess/pipeline.js";
import { RGBImage } from "../src/types.js";

/**
 * Tests for the fused preprocessing path a classifier takes.
 *
 * The resampling itself is a canvas `drawImage`, which Node has none of — and
 * it is byte for byte the same call the composable `resize()` makes, so there
 * is nothing to compare there anyway. What *is* worth proving in Node is the
 * arithmetic that replaced `normalize` → `toCHW`: same values, same rounding,
 * written straight into a reused buffer.
 *
 * That is the claim the 0.6.0 letterbox work made about its own output, and the
 * one a caller depends on when a model's predictions are compared across SDK
 * versions.
 */

/** Deterministic pseudo-random image, so a failure reproduces exactly. */
function image(width: number, height: number, seed = 1): RGBImage {
  const data = new Uint8Array(width * height * 3);
  let state = seed;
  for (let i = 0; i < data.length; i++) {
    state = (state * 1103515245 + 12345) % 2147483648;
    data[i] = state % 256;
  }
  return new RGBImage(data, width, height);
}

/** The composable path a classifier took before the pipeline existed. */
function composedPath(
  source: RGBImage,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
): Float32Array {
  return toCHW(normalize(source, mean, std), source.width, source.height, 3);
}

describe("writePlanarFloat32", () => {
  const cases: Array<{
    name: string;
    mean: readonly [number, number, number];
    std: readonly [number, number, number];
  }> = [
    { name: "ImageNet mean/std", mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
    { name: "Ultralytics pixel/255", mean: [0, 0, 0], std: [1, 1, 1] },
    { name: "asymmetric per-channel", mean: [0.1, 0.5, 0.9], std: [0.3, 1.7, 0.05] },
  ];

  for (const { name, mean, std } of cases) {
    it(`is bit-identical to normalize + toCHW — ${name}`, () => {
      const source = image(7, 5);
      const out = new Float32Array(3 * source.width * source.height);

      writePlanarFloat32(source.data, source.width, source.height, mean, std, out, 3);

      expect(Array.from(out)).toEqual(Array.from(composedPath(source, mean, std)));
    });
  }

  it("reads RGBA with the default stride, skipping alpha", () => {
    const rgba = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 128]);
    const out = new Float32Array(6);

    writePlanarFloat32(rgba, 2, 1, [0, 0, 0], [1, 1, 1], out);

    expect(Array.from(out)).toEqual(
      Array.from(new Float32Array([10 / 255, 40 / 255, 20 / 255, 50 / 255, 30 / 255, 60 / 255])),
    );
  });
});

describe("ResizePipeline", () => {
  it("passes an already-sized image through without a canvas", () => {
    const source = image(8, 8);
    const pipeline = new ResizePipeline(8, 8, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);

    const { data } = pipeline.run(source);

    expect(Array.from(data)).toEqual(
      Array.from(composedPath(source, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])),
    );
  });

  it("reuses one buffer across calls, and stops once a result is checked out", () => {
    const pipeline = new ResizePipeline(4, 4);

    const first = pipeline.run(image(4, 4));
    const second = pipeline.run(image(4, 4, 9));

    expect(first.reused).toBe(true);
    expect(second.reused).toBe(false);
    expect(second.data).not.toBe(first.data);

    pipeline.release();
    expect(pipeline.run(image(4, 4)).reused).toBe(true);
  });

  it("reports the target size it was built for", () => {
    expect(new ResizePipeline(224, 160).targetSize).toEqual([224, 160]);
  });

  it("rejects a non-positive target", () => {
    expect(() => new ResizePipeline(0, 224)).toThrow(/Invalid resize target/);
    expect(() => new ResizePipeline(224, -1)).toThrow(/Invalid resize target/);
  });

  it("has a one-shot form", () => {
    const source = image(6, 6);

    expect(Array.from(resizeToTensorData(source, 6, 6).data)).toEqual(
      Array.from(composedPath(source, [0, 0, 0], [1, 1, 1])),
    );
  });
});

describe("zeroTensorData", () => {
  it("sizes the warm-up payload for a CHW input", () => {
    expect(zeroTensorData(224, 224).length).toBe(3 * 224 * 224);
    expect(zeroTensorData(4, 2).every((value) => value === 0)).toBe(true);
  });
});
