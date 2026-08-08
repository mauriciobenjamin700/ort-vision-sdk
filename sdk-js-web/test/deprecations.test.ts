/**
 * Guards that the removed `decodeYoloV8*` aliases stay removed.
 *
 * They were deprecated in 0.2.0 with "will be removed in 0.4.0" and were still
 * shipping at 0.5.1, which is how a deprecation stops meaning anything. Now
 * that they are gone, this file exists so a well-meaning re-add — or a merge
 * that resurrects them — fails loudly instead of quietly restoring names the
 * changelog says do not exist.
 *
 * The replacements are the same functions under honest names: the decoder was
 * never v8-specific, it covers every anchor-free YOLO head from v8 to v12.
 */

import { describe, expect, it } from "vitest";

import * as sdk from "../src/index.js";
import * as detection from "../src/postprocess/detection.js";
import * as segmentation from "../src/postprocess/segmentation.js";

/** Removed alias → the name that replaced it. */
const REMOVED: Record<string, string> = {
  decodeYoloV8: "decodeYolo",
  decodeYoloV8Anchors: "decodeYoloAnchors",
  decodeYoloV8Seg: "decodeYoloSeg",
};

describe("removed decodeYoloV8* aliases", () => {
  for (const [removed, replacement] of Object.entries(REMOVED)) {
    it(`${removed} is gone from the package entry point`, () => {
      expect(removed in sdk).toBe(false);
    });

    it(`${removed} is gone from its own module`, () => {
      expect(removed in detection).toBe(false);
      expect(removed in segmentation).toBe(false);
    });

    it(`${replacement} replaced it and is still exported`, () => {
      expect(typeof (sdk as Record<string, unknown>)[replacement]).toBe("function");
    });
  }
});
