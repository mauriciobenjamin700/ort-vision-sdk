/**
 * `modelHasHalfWeights`: telling a half-precision export apart from the
 * outside, including one whose inputs and outputs stayed float32 — the case
 * the declared input types cannot see.
 */

import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { modelHasHalfWeights } from "../src/core/metadata.js";

const MODELS = new URL("../../sdk-python/tests/fixtures/models/", import.meta.url);

/** Read one of the shared fixture models into memory. */
function model(name: string): Uint8Array {
  return new Uint8Array(readFileSync(new URL(name, MODELS)));
}

/** Encode a protobuf varint. */
function varint(value: number): number[] {
  const out: number[] = [];
  let rest = value;
  while (rest > 0x7f) {
    out.push((rest & 0x7f) | 0x80);
    rest = Math.floor(rest / 128);
  }
  out.push(rest);
  return out;
}

/** Encode a length-delimited field. */
function field(number: number, payload: readonly number[]): number[] {
  return [...varint((number << 3) | 2), ...varint(payload.length), ...payload];
}

/**
 * A `ModelProto` whose graph holds initializers of the given data types.
 *
 * Each tensor carries `dims` before `data_type` and a raw payload after it, so
 * the reader has to skip fields on both sides of the one it wants.
 */
function withInitializers(dataTypes: readonly number[]): Uint8Array {
  const graph: number[] = [...field(1, [0x6e])];
  for (const dataType of dataTypes) {
    const tensor = [8, 2, 16, ...varint(dataType), ...field(9, [0, 0, 0, 0])];
    graph.push(...field(5, tensor));
  }
  return new Uint8Array(field(7, graph));
}

describe("modelHasHalfWeights", () => {
  it("finds a float16 initializer among float32 ones", () => {
    expect(modelHasHalfWeights(withInitializers([1, 1, 10]))).toBe(true);
  });

  it("is false for a graph of float32 initializers", () => {
    expect(modelHasHalfWeights(withInitializers([1, 7, 1]))).toBe(false);
  });

  it("is false for a file with no graph", () => {
    expect(modelHasHalfWeights(new Uint8Array([]))).toBe(false);
  });

  it("is false for a truncated file instead of throwing", () => {
    const bytes = withInitializers([10]);
    expect(modelHasHalfWeights(bytes.subarray(0, 5))).toBe(false);
  });

  it("finds a float16 tensor held by a Constant node", () => {
    const tensor = [8, 2, 16, 10, ...field(9, [0, 0])];
    const attribute = [...field(1, [0x76]), ...field(5, tensor)];
    const node = [...field(4, [0x43]), ...field(5, attribute)];
    expect(modelHasHalfWeights(new Uint8Array(field(7, field(1, node))))).toBe(true);
  });

  it("tells the shared half fixture from its float32 twin", () => {
    expect(modelHasHalfWeights(model("tiny_classifier_fp16.onnx"))).toBe(true);
    expect(modelHasHalfWeights(model("tiny_classifier.onnx"))).toBe(false);
  });
});
