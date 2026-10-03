/**
 * Declared shapes must not depend on which `onnxruntime-web` is installed.
 *
 * The tasks read their input size and class count off the shapes a session
 * declares. `onnxruntime-web` reports those only from 1.22 on; below that the
 * session answered nothing and every task fell back to its defaults, so a 64x64
 * detector was fed 640x640 and ORT aborted the run with
 * `Got invalid dimensions for input` (measured on 1.17.3, 1.18.0, 1.19.2, 1.20.1 and 1.21.0). These
 * tests pin the file-side reader that closes the gap and the session's fallback
 * to it.
 */

import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import { readModelShapes } from "../src/core/metadata.js";

const createSession = vi.fn((_model: unknown, _options?: unknown) =>
  Promise.resolve({
    inputNames: ["images"],
    outputNames: ["output0", "output1"],
    inputMetadata: undefined,
    outputMetadata: undefined,
    release: () => Promise.resolve(),
  }),
);

vi.mock("onnxruntime-web", () => ({
  InferenceSession: {
    get create() {
      return createSession;
    },
  },
}));

const { OrtSession } = await import("../src/core/session.js");

const MODELS = new URL("../../sdk-python/tests/fixtures/models/", import.meta.url);

/**
 * Read a fixture model shared with the Python suite.
 *
 * @param name File name under the fixtures directory.
 * @returns The model bytes.
 */
function model(name: string): Uint8Array {
  return new Uint8Array(readFileSync(new URL(name, MODELS)));
}

/** Protobuf wire types, for the hand-built models below. */
const WIRE_VARINT = 0;
const WIRE_LENGTH_DELIMITED = 2;

/**
 * Encode a base-128 varint.
 *
 * @param value The number to encode.
 * @returns Its bytes.
 */
function varint(value: number): number[] {
  const bytes: number[] = [];
  let rest = value;
  while (rest > 0x7f) {
    bytes.push((rest & 0x7f) | 0x80);
    rest = Math.floor(rest / 128);
  }
  bytes.push(rest);
  return bytes;
}

/**
 * Encode a length-delimited field: tag, length, payload.
 *
 * @param field Field number.
 * @param payload The field's bytes.
 * @returns The encoded field.
 */
function delimited(field: number, payload: readonly number[]): number[] {
  return [...varint(field * 8 + WIRE_LENGTH_DELIMITED), ...varint(payload.length), ...payload];
}

/**
 * Encode a string field.
 *
 * @param field Field number.
 * @param text The string.
 * @returns The encoded field.
 */
function text(field: number, text: string): number[] {
  return delimited(field, [...new TextEncoder().encode(text)]);
}

/**
 * Encode a `ValueInfoProto` for a float tensor.
 *
 * @param name The value's name.
 * @param dims Each dimension: a number for `dim_value`, a string for a
 *   symbolic `dim_param`.
 * @returns The encoded message.
 */
function valueInfo(name: string, dims: readonly (number | string)[]): number[] {
  const shape = dims.flatMap((dim) =>
    delimited(1, typeof dim === "number" ? [...varint(1 * 8 + WIRE_VARINT), ...varint(dim)] : text(2, dim)),
  );
  const tensor = [...varint(1 * 8 + WIRE_VARINT), ...varint(1), ...delimited(2, shape)];
  return [...text(1, name), ...delimited(2, delimited(1, tensor))];
}

/**
 * Encode a `ModelProto` whose graph declares the given inputs and outputs.
 *
 * @param inputs `GraphProto.input` entries.
 * @param outputs `GraphProto.output` entries.
 * @returns The encoded model.
 */
function modelProto(inputs: readonly number[][], outputs: readonly number[][]): Uint8Array {
  const graph = [...inputs.flatMap((v) => delimited(11, v)), ...outputs.flatMap((v) => delimited(12, v))];
  return new Uint8Array(delimited(7, graph));
}

describe("readModelShapes", () => {
  it("reads what ORT reports for each fixture", () => {
    expect(readModelShapes(model("tiny_detector.onnx"))).toEqual({
      inputs: { images: [1, 3, 64, 64] },
      outputs: { output0: [1, 7, 20] },
    });
    expect(readModelShapes(model("tiny_classifier.onnx"))).toEqual({
      inputs: { images: [1, 3, 32, 32] },
      outputs: { logits: [1, 4] },
    });
    expect(readModelShapes(model("tiny_segmenter.onnx"))).toEqual({
      inputs: { images: [1, 3, 64, 64] },
      outputs: { output0: [1, 39, 40], output1: [1, 32, 8, 8] },
    });
  });

  it("reads a symbolic dimension as null", () => {
    const bytes = modelProto([valueInfo("images", ["batch", 3, 640, 640])], [valueInfo("out", ["batch", 84, 8400])]);

    expect(readModelShapes(bytes)).toEqual({
      inputs: { images: [null, 3, 640, 640] },
      outputs: { out: [null, 84, 8400] },
    });
  });

  it("keys by name, so an initializer listed as an input does not shift the real one", () => {
    const bytes = modelProto([valueInfo("conv.weight", [16, 3, 3, 3]), valueInfo("images", [1, 3, 64, 64])], []);

    expect(readModelShapes(bytes).inputs["images"]).toEqual([1, 3, 64, 64]);
  });

  it("returns nothing for bytes that are not a model", () => {
    expect(readModelShapes(new Uint8Array([0, 1, 2, 3]))).toEqual({ inputs: {}, outputs: {} });
  });
});

describe("OrtSession on a runtime that reports no shapes", () => {
  it("falls back to the shapes the file declares", async () => {
    const session = await OrtSession.create(model("tiny_segmenter.onnx"), { providers: ["wasm"] });

    expect(session.inputShape).toEqual([1, 3, 64, 64]);
    expect(session.outputShapes).toEqual([
      [1, 39, 40],
      [1, 32, 8, 8],
    ]);
  });

  it("declares nothing when the bytes were never read", async () => {
    const session = await OrtSession.create("/models/remote.onnx", {
      providers: ["wasm"],
      readMetadata: false,
    });

    expect(session.inputShapes).toEqual([[]]);
    expect(session.outputShapes).toEqual([[], []]);
  });
});
