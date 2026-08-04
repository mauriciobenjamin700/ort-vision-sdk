import { describe, expect, it } from "vitest";

import { modelNames, readModelMetadata } from "../src/core/metadata.js";

/** Field number of `metadata_props` in `ModelProto`. */
const METADATA_PROPS_FIELD = 14;

/**
 * Encode a base-128 varint.
 *
 * @param value Non-negative integer to encode.
 * @returns Its varint bytes.
 */
function varint(value: number): number[] {
  const out: number[] = [];
  let remaining = value;
  while (remaining > 0x7f) {
    out.push((remaining & 0x7f) | 0x80);
    remaining = Math.floor(remaining / 128);
  }
  out.push(remaining);
  return out;
}

/**
 * Encode a length-delimited field: tag, length, payload.
 *
 * @param field Field number.
 * @param payload Bytes of the field's value.
 * @returns The encoded field.
 */
function lengthDelimited(field: number, payload: readonly number[]): number[] {
  return [...varint((field << 3) | 2), ...varint(payload.length), ...payload];
}

/**
 * Build a minimal `ModelProto` carrying only a metadata map.
 *
 * Mirrors what an exporter writes, so the reader is exercised against the real
 * wire format rather than a stand-in.
 *
 * @param entries Metadata key/value pairs.
 * @param extras Extra raw bytes to prepend, standing for the fields a real
 *   model has before its metadata (graph, opset, producer, ...).
 * @returns The encoded model bytes.
 */
function modelProto(
  entries: Readonly<Record<string, string>>,
  extras: readonly number[] = [],
): Uint8Array {
  const encoder = new TextEncoder();
  const bytes: number[] = [...extras];
  for (const [key, value] of Object.entries(entries)) {
    const entry = [
      ...lengthDelimited(1, [...encoder.encode(key)]),
      ...lengthDelimited(2, [...encoder.encode(value)]),
    ];
    bytes.push(...lengthDelimited(METADATA_PROPS_FIELD, entry));
  }
  return new Uint8Array(bytes);
}

describe("readModelMetadata", () => {
  it("reads the metadata map an exporter wrote", () => {
    const model = modelProto({
      task: "classify",
      imgsz: "[224, 224]",
      names: "{0: 'deworm', 1: 'not_deworm'}",
    });

    expect(readModelMetadata(model)).toEqual({
      task: "classify",
      imgsz: "[224, 224]",
      names: "{0: 'deworm', 1: 'not_deworm'}",
    });
  });

  it("skips the other ModelProto fields to reach the metadata", () => {
    const producer = lengthDelimited(2, [...new TextEncoder().encode("pytorch")]);
    const irVersion = [...varint((1 << 3) | 0), ...varint(9)];
    const opset = [...varint((8 << 3) | 5), 0, 0, 0, 0];

    const model = modelProto({ names: "{0: 'a'}" }, [...irVersion, ...producer, ...opset]);

    expect(readModelMetadata(model)).toEqual({ names: "{0: 'a'}" });
  });

  it("accepts an ArrayBuffer as well as a view", () => {
    const model = modelProto({ task: "detect" });
    expect(readModelMetadata(model.buffer as ArrayBuffer)).toEqual({ task: "detect" });
  });

  it("returns an empty map for junk instead of throwing", () => {
    expect(readModelMetadata(new Uint8Array([0xff, 0xff, 0xff, 0xff]))).toEqual({});
    expect(readModelMetadata(new Uint8Array())).toEqual({});
    // A metadata entry whose declared length runs past the buffer.
    expect(readModelMetadata(new Uint8Array([(METADATA_PROPS_FIELD << 3) | 2, 200, 1, 2]))).toEqual(
      {},
    );
  });
});

describe("modelNames", () => {
  it("parses the dict repr Ultralytics writes", () => {
    expect(modelNames({ names: "{0: 'deworm', 1: 'not_deworm'}" })).toEqual([
      "deworm",
      "not_deworm",
    ]);
    expect(modelNames({ names: "{0: 'ocular-mucosa'}" })).toEqual(["ocular-mucosa"]);
    expect(modelNames({ names: '{0: "person", 1: "bicycle"}' })).toEqual(["person", "bicycle"]);
  });

  it("resolves the escapes a repr emits", () => {
    expect(modelNames({ names: "{0: 'it\\'s', 1: 'a\\\\b'}" })).toEqual(["it's", "a\\b"]);
  });

  it("rejects a map that is not keyed by contiguous ids from zero", () => {
    expect(modelNames({ names: "{1: 'a', 2: 'b'}" })).toBeNull();
    expect(modelNames({ names: "{0: 'a', 2: 'b'}" })).toBeNull();
    expect(modelNames({ names: "{-1: 'a'}" })).toBeNull();
  });

  it("returns null when there is nothing usable to read", () => {
    expect(modelNames(undefined)).toBeNull();
    expect(modelNames({})).toBeNull();
    expect(modelNames({ names: "" })).toBeNull();
    expect(modelNames({ names: "{}" })).toBeNull();
    expect(modelNames({ names: "['a', 'b']" })).toBeNull();
    expect(modelNames({ names: "not a dict" })).toBeNull();
  });

  it("rejects a dict carrying content it could not account for", () => {
    expect(modelNames({ names: "{0: 'a', 1: unquoted}" })).toBeNull();
  });
});
