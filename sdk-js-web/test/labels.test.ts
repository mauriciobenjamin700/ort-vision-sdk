import { describe, expect, it } from "vitest";

import { LabelMapError } from "../src/core/exceptions.js";
import { COCO_CLASSES, resolveLabels } from "../src/labels.js";

describe("COCO_CLASSES preset", () => {
  it("has 80 classes in canonical Ultralytics order", () => {
    expect(COCO_CLASSES.length).toBe(80);
    expect(COCO_CLASSES[0]).toBe("person");
    expect(COCO_CLASSES[16]).toBe("dog");
    expect(COCO_CLASSES[79]).toBe("toothbrush");
  });
});

describe("resolveLabels", () => {
  it("resolves the 'coco' preset", () => {
    expect(resolveLabels("coco")).toEqual(COCO_CLASSES);
  });

  it("throws on unknown preset", () => {
    expect(() => resolveLabels("imagenet")).toThrow(LabelMapError);
  });

  it("returns explicit array unchanged", () => {
    const labels = ["a", "b", "c"];
    expect(resolveLabels(labels)).toEqual(["a", "b", "c"]);
  });

  it("validates length against numClasses", () => {
    expect(() => resolveLabels(["a", "b"], { numClasses: 5 })).toThrow(LabelMapError);
  });

  it("expands sparse dict, filling gaps with class_<id>", () => {
    const result = resolveLabels({ 0: "first", 2: "third" });
    expect(result).toEqual(["first", "class_1", "third"]);
  });

  it("auto-generates class_X labels when spec is null + numClasses given", () => {
    expect(resolveLabels(null, { numClasses: 3 })).toEqual(["class_0", "class_1", "class_2"]);
  });

  it("auto-generates from undefined the same way", () => {
    expect(resolveLabels(undefined, { numClasses: 2 })).toEqual(["class_0", "class_1"]);
  });

  it("throws when null + numClasses missing", () => {
    expect(() => resolveLabels(null)).toThrow(LabelMapError);
  });
});
