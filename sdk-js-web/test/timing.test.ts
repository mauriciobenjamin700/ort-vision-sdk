import { afterEach, describe, expect, it, vi } from "vitest";

import { SpeedTimer } from "../src/core/timing.js";
import { ClassificationResults, DetectionResults, Probs, Boxes } from "../src/results.js";
import { RGBImage } from "../src/types.js";

/** Drive `performance.now()` from a script so durations are exact. */
function stubClock(sequence: number[]): void {
  let index = 0;
  vi.spyOn(performance, "now").mockImplementation(() => {
    const value = sequence[Math.min(index, sequence.length - 1)] as number;
    index += 1;
    return value;
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("SpeedTimer", () => {
  it("attributes each interval to the stage that closed it", () => {
    stubClock([0, 10, 25, 125, 150]);

    const timer = new SpeedTimer();
    timer.stage("load");
    timer.stage("preprocess");
    timer.stage("inference");
    timer.stage("postprocess");

    expect(timer.speed()).toEqual({
      load: 10,
      preprocess: 15,
      inference: 100,
      postprocess: 25,
    });
  });

  it("accumulates when a stage is closed more than once", () => {
    stubClock([0, 5, 8, 20]);

    const timer = new SpeedTimer();
    timer.stage("inference");
    timer.stage("postprocess");
    timer.stage("inference");

    expect(timer.speed().inference).toBe(17);
    expect(timer.speed().postprocess).toBe(3);
  });

  it("starts every stage at zero", () => {
    stubClock([0]);

    expect(new SpeedTimer().speed()).toEqual({
      load: 0,
      preprocess: 0,
      inference: 0,
      postprocess: 0,
    });
  });

  it("returns a copy so later stages do not mutate a captured snapshot", () => {
    stubClock([0, 4, 9]);

    const timer = new SpeedTimer();
    timer.stage("load");
    const snapshot = timer.speed();
    timer.stage("load");

    expect(snapshot.load).toBe(4);
    expect(timer.speed().load).toBe(9);
  });
});

describe("Results envelopes", () => {
  const image = new RGBImage(new Uint8Array(3), 1, 1);

  it("default to zero timings when built outside predict()", () => {
    const envelope = new DetectionResults(
      new Boxes(new Float32Array(0), new Int32Array(0), new Float32Array(0), [1, 1]),
      [],
      {},
      image,
      [1, 1],
    );

    expect(envelope.speed).toEqual({
      load: 0,
      preprocess: 0,
      inference: 0,
      postprocess: 0,
    });
  });

  it("carry the timings they are given", () => {
    const speed = { load: 1, preprocess: 2, inference: 3, postprocess: 4 };
    const envelope = new ClassificationResults(
      new Probs(new Float32Array([1])),
      {
        classId: 0,
        className: "a",
        confidence: 1,
        cls: 0,
        name: "a",
        conf: 1,
        image,
        probabilities: [],
      },
      { 0: "a" },
      image,
      [1, 1],
      null,
      speed,
    );

    expect(envelope.speed).toEqual(speed);
  });
});
