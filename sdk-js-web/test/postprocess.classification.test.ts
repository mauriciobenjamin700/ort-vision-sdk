import { describe, expect, it } from "vitest";

import { softmax, topK } from "../src/postprocess/classification.js";

describe("softmax", () => {
  it("sums to 1", () => {
    const probs = softmax(new Float32Array([2, 1, 0.1, 5, -1]));
    const sum = Array.from(probs).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it("preserves argmax", () => {
    const logits = new Float32Array([0.1, 0.5, 5.0, 0.2, 0.3]);
    const probs = softmax(logits);
    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }
    expect(maxIdx).toBe(2);
  });

  it("is numerically stable with large logits", () => {
    const probs = softmax(new Float32Array([100, -100, 0, 50]));
    expect(probs.every((v) => Number.isFinite(v))).toBe(true);
    expect(probs.every((v) => v >= 0 && v <= 1)).toBe(true);
  });

  it("accepts plain number arrays", () => {
    const probs = softmax([1, 2, 3]);
    expect(probs.length).toBe(3);
    expect(Array.from(probs).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
  });
});

describe("topK", () => {
  it("returns top-k descending", () => {
    const probs = new Float32Array([0.05, 0.1, 0.5, 0.25, 0.1]);
    const { indices, values } = topK(probs, 3);
    expect(Array.from(indices)).toEqual([2, 3, 1]);
    expect(values[0]).toBeCloseTo(0.5);
    expect(values[1]).toBeCloseTo(0.25);
  });

  it("k=null returns full sorted vector", () => {
    const { indices } = topK(new Float32Array([0.3, 0.5, 0.2]), null);
    expect(Array.from(indices)).toEqual([1, 0, 2]);
  });

  it("clamps k to length when k > N", () => {
    const { indices, values } = topK(new Float32Array([0.6, 0.4]), 10);
    expect(indices.length).toBe(2);
    expect(values.length).toBe(2);
  });

  it("returns empty arrays for k=0", () => {
    const { indices, values } = topK(new Float32Array([0.5, 0.3]), 0);
    expect(indices.length).toBe(0);
    expect(values.length).toBe(0);
  });
});
