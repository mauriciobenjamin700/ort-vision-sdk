import { describe, expect, it } from "vitest";

import { METADATA_PREFIX, readFusionSpec } from "../src/fusion.js";

/**
 * Tests for the metadata contract a fused pipeline carries inside its own file.
 *
 * This is the browser half of a contract whose other half is written in Python
 * by `ort_vision_sdk.compose`. The encodings tested here are literally the
 * strings that end up in the `.onnx`, so `sdk-python/tests/test_fusion.py` and
 * this file must agree case for case — a divergence would show up as a pipeline
 * that runs at one resolution in Python and another in the browser.
 */

/**
 * Build the `ovs.*` metadata a fusion would have written.
 *
 * @param overrides Entries to replace or add, keys given without the prefix.
 * @returns The metadata map.
 */
function metadata(overrides: Readonly<Record<string, string>> = {}): Record<string, string> {
  const base: Record<string, string> = {
    kind: "detect_classify",
    sdk_version: "9.9.9",
    input_size: "640,640",
    crop_size: "224,224",
    crop_source: "detector_input",
    max_detections: "20",
    conf_threshold: "0.25",
    iou_threshold: "0.45",
    apply_softmax: "1",
    detector_names: "{0: 'sheep'}",
    classifier_names: "{0: 'healthy', 1: 'anaemic'}",
    ...overrides,
  };
  const prefixed: Record<string, string> = {};
  for (const [key, value] of Object.entries(base)) prefixed[`${METADATA_PREFIX}${key}`] = value;
  return prefixed;
}

describe("readFusionSpec", () => {
  it("decodes every field a fusion records", () => {
    const spec = readFusionSpec(metadata());

    expect(spec).not.toBeNull();
    expect(spec?.inputSize).toEqual([640, 640]);
    expect(spec?.cropSize).toEqual([224, 224]);
    expect(spec?.cropSource).toBe("detector_input");
    expect(spec?.maxDetections).toBe(20);
    expect(spec?.confThreshold).toBeCloseTo(0.25);
    expect(spec?.iouThreshold).toBeCloseTo(0.45);
    expect(spec?.applySoftmax).toBe(true);
    expect(spec?.detectorNames).toEqual(["sheep"]);
    expect(spec?.classifierNames).toEqual(["healthy", "anaemic"]);
    expect(spec?.sdkVersion).toBe("9.9.9");
  });

  it("reads the dynamic row mode", () => {
    expect(readFusionSpec(metadata({ max_detections: "dynamic" }))?.maxDetections).toBeNull();
  });

  it("flags a pipeline that needs the full-resolution input", () => {
    const spec = readFusionSpec(metadata({ crop_source: "original" }));

    expect(spec?.cropSource).toBe("original");
    expect(spec?.needsSourceImage).toBe(true);
  });

  it("leaves absent class maps null rather than inventing empty ones", () => {
    const entries = metadata();
    delete entries[`${METADATA_PREFIX}detector_names`];
    delete entries[`${METADATA_PREFIX}classifier_names`];

    const spec = readFusionSpec(entries);
    expect(spec?.detectorNames).toBeNull();
    expect(spec?.classifierNames).toBeNull();
  });

  it("ignores the exporter's own metadata sitting alongside ours", () => {
    const spec = readFusionSpec({
      ...metadata(),
      names: "{0: 'something else'}",
      task: "detect",
    });

    expect(spec?.detectorNames).toEqual(["sheep"]);
  });
});

describe("readFusionSpec rejection", () => {
  it("returns null for no metadata at all", () => {
    expect(readFusionSpec(undefined)).toBeNull();
    expect(readFusionSpec({})).toBeNull();
  });

  it("returns null for a plain model", () => {
    expect(readFusionSpec({ names: "{0: 'cat'}", task: "detect" })).toBeNull();
  });

  it("returns null for a pipeline kind this version cannot drive", () => {
    expect(readFusionSpec(metadata({ kind: "detect_segment_classify" }))).toBeNull();
  });

  it.each(["input_size", "crop_size"])(
    "returns null when %s is unreadable, since a resolution has no safe default",
    (key) => {
      expect(readFusionSpec(metadata({ [key]: "not-a-size" }))).toBeNull();
    },
  );

  it.each(["640", "640,0", "640,640,640", ""])(
    "returns null for the malformed size %o",
    (value) => {
      expect(readFusionSpec(metadata({ input_size: value }))).toBeNull();
    },
  );
});

describe("readFusionSpec tolerance", () => {
  it("falls back on a malformed threshold instead of rejecting the pipeline", () => {
    const spec = readFusionSpec(metadata({ conf_threshold: "quite high" }));

    expect(spec).not.toBeNull();
    expect(spec?.confThreshold).toBeCloseTo(0.25);
  });

  it("falls back to the dynamic mode on a malformed row count", () => {
    const spec = readFusionSpec(metadata({ max_detections: "several" }));

    expect(spec).not.toBeNull();
    expect(spec?.maxDetections).toBeNull();
  });

  it("falls back to the single-input crop source on an unrecognised one", () => {
    const spec = readFusionSpec(metadata({ crop_source: "somewhere_else" }));

    expect(spec?.cropSource).toBe("detector_input");
    expect(spec?.needsSourceImage).toBe(false);
  });

  it("drops a class map keyed by non-contiguous ids rather than half-applying it", () => {
    const spec = readFusionSpec(
      metadata({ classifier_names: "{0: 'healthy', 7: 'anaemic'}" }),
    );

    expect(spec).not.toBeNull();
    expect(spec?.classifierNames).toBeNull();
  });

  it("honours a classifier that already normalizes its output", () => {
    expect(readFusionSpec(metadata({ apply_softmax: "0" }))?.applySoftmax).toBe(false);
  });
});
