/**
 * Microbenchmarks for the web SDK's hot pure functions.
 *
 * The web counterpart of `scripts/bench.py`: same cases, same shapes, same
 * thresholds, so a number here can be read next to its Python twin. Only what
 * runs without a DOM is measured — decode, NMS, mask assembly and the pixel
 * conversion loops (with a stand-in `ImageData`). Canvas work (`drawImage`, `getImageData`) needs a real
 * browser and is not covered.
 *
 * It imports the built package, so build first:
 *
 *   npm --prefix sdk-js-web run build
 *   node scripts/bench_web.mjs
 *   node scripts/bench_web.mjs --json bench/baseline-web.json
 *   node scripts/bench_web.mjs --compare bench/baseline-web.json
 *
 * Like the Python baseline, the committed file is a local reference, not a CI
 * gate: run it on the machine that recorded it, before and after a change.
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import os from "node:os";

import { imageDataToRGBChecked, rgbToImageData } from "../sdk-js-web/dist/core/canvas.js";
import { batchedNms, decodeYolo, nms } from "../sdk-js-web/dist/postprocess/detection.js";
import { decodeYoloSeg } from "../sdk-js-web/dist/postprocess/segmentation.js";
import { writePlanarFloat32 } from "../sdk-js-web/dist/preprocess/pipeline.js";
import { RGBImage } from "../sdk-js-web/dist/types.js";

/**
 * Minimal `ImageData`, which Node lacks and `rgbToImageData` constructs.
 *
 * The conversions only touch it when called, so defining it after the imports
 * is enough for them to run their real code.
 */
globalThis.ImageData ??= class ImageData {
  constructor(data, width, height) {
    this.data = data;
    this.width = width;
    this.height = height;
  }
};

const NUM_CLASSES = 80;
const NUM_ANCHORS = 8400;
const NUM_MASK_COEFS = 32;
const MASK_SIZE = 160;
const DEFAULT_TOLERANCE = 0.25;
const NOISE_FLOOR_MS = 0.1;

/**
 * A seeded generator, so every case sees identical inputs across runs.
 *
 * @returns A function yielding uniform floats in `[0, 1)`.
 */
function rng() {
  let state = 20260806;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Build a detect tensor with a chosen number of above-threshold anchors.
 *
 * @param {number} numCandidates Anchors carrying a class score of 0.9.
 * @returns {Float32Array} `(4 + NUM_CLASSES) * NUM_ANCHORS` floats.
 */
function makeDetectionOutput(numCandidates) {
  const rand = rng();
  const out = new Float32Array((4 + NUM_CLASSES) * NUM_ANCHORS);
  for (let i = 4 * NUM_ANCHORS; i < out.length; i++) out[i] = rand() * 0.2;
  for (let a = 0; a < NUM_ANCHORS; a++) {
    out[a] = rand() * 640;
    out[NUM_ANCHORS + a] = rand() * 640;
    out[2 * NUM_ANCHORS + a] = 10 + rand() * 110;
    out[3 * NUM_ANCHORS + a] = 10 + rand() * 110;
  }
  const anchors = Array.from({ length: NUM_ANCHORS }, (_, i) => i);
  for (let i = anchors.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [anchors[i], anchors[j]] = [anchors[j], anchors[i]];
  }
  for (let i = 0; i < numCandidates; i++) {
    out[(4 + (i % NUM_CLASSES)) * NUM_ANCHORS + anchors[i]] = 0.9;
  }
  return out;
}

/**
 * Build per-anchor and prototype tensors for a seg head.
 *
 * @param {number} numInstances Anchors carrying an above-threshold score.
 * @returns {{ perAnchor: Float32Array, prototypes: Float32Array }}
 */
function makeSegmentationOutputs(numInstances) {
  const rand = rng();
  const normal = () => {
    const u = Math.max(rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const channels = 4 + NUM_CLASSES + NUM_MASK_COEFS;
  const perAnchor = new Float32Array(channels * NUM_ANCHORS);
  for (let i = 4 * NUM_ANCHORS; i < (4 + NUM_CLASSES) * NUM_ANCHORS; i++) {
    perAnchor[i] = rand() * 0.2;
  }
  for (let i = 0; i < numInstances; i++) {
    const a = i * Math.floor(NUM_ANCHORS / numInstances);
    perAnchor[a] = 100 + rand() * 440;
    perAnchor[NUM_ANCHORS + a] = 100 + rand() * 440;
    perAnchor[2 * NUM_ANCHORS + a] = 80;
    perAnchor[3 * NUM_ANCHORS + a] = 80;
    perAnchor[(4 + (i % NUM_CLASSES)) * NUM_ANCHORS + a] = 0.9;
    for (let k = 0; k < NUM_MASK_COEFS; k++) {
      perAnchor[(4 + NUM_CLASSES + k) * NUM_ANCHORS + a] = normal() * 0.5;
    }
  }
  const prototypes = new Float32Array(NUM_MASK_COEFS * MASK_SIZE * MASK_SIZE);
  for (let i = 0; i < prototypes.length; i++) prototypes[i] = normal();
  return { perAnchor, prototypes };
}

/**
 * Build overlapping boxes, scores and class ids for the NMS cases.
 *
 * @param {number} count Number of boxes.
 */
function makeBoxes(count) {
  const rand = rng();
  const boxes = new Float32Array(count * 4);
  const scores = new Float32Array(count);
  const classIds = new Int32Array(count);
  for (let i = 0; i < count; i++) {
    const cx = 50 + rand() * 540;
    const cy = 50 + rand() * 540;
    const side = 20 + rand() * 70;
    boxes[i * 4] = cx - side / 2;
    boxes[i * 4 + 1] = cy - side / 2;
    boxes[i * 4 + 2] = cx + side / 2;
    boxes[i * 4 + 3] = cy + side / 2;
    scores[i] = 0.25 + rand() * 0.75;
    classIds[i] = Math.floor(rand() * 20);
  }
  return { boxes, scores, classIds };
}

/**
 * Build a packed RGB image.
 *
 * @param {number} width Width in pixels.
 * @param {number} height Height in pixels.
 */
function makeImage(width, height) {
  const rand = rng();
  const data = new Uint8Array(width * height * 3);
  for (let i = 0; i < data.length; i++) data[i] = Math.floor(rand() * 256);
  return new RGBImage(data, width, height);
}

/** Assemble every case as a zero-argument callable, inputs built up front. */
function buildCases() {
  const emptyOutput = makeDetectionOutput(1);
  const sparseOutput = makeDetectionOutput(50);
  const denseOutput = makeDetectionOutput(2000);
  const { perAnchor, prototypes } = makeSegmentationOutputs(30);
  const b300 = makeBoxes(300);
  const b2000 = makeBoxes(2000);
  const hd = makeImage(1920, 1080);
  const hdImageData = rgbToImageData(hd);
  const letterboxed = new Uint8ClampedArray(640 * 640 * 4).fill(114);
  const planar = new Float32Array(3 * 640 * 640);
  const dims = [1, 4 + NUM_CLASSES, NUM_ANCHORS];

  const decodeOptions = {
    originalWidth: 1920,
    originalHeight: 1080,
    padLeft: 0,
    padTop: 140,
    scale: 640 / 1920,
    confThreshold: 0.25,
    iouThreshold: 0.45,
    maxDetections: 300,
  };

  return {
    rgba_to_rgb_1080p: () => imageDataToRGBChecked(hdImageData),
    rgb_to_rgba_1080p: () => rgbToImageData(hd),
    write_planar_float32_640: () =>
      writePlanarFloat32(letterboxed, 640, 640, [0, 0, 0], [1, 1, 1], planar),
    decode_yolo_no_candidates: () =>
      decodeYolo(emptyOutput, dims, { ...decodeOptions, confThreshold: 0.95 }),
    decode_yolo_50_candidates: () => decodeYolo(sparseOutput, dims, decodeOptions),
    decode_yolo_2000_candidates: () => decodeYolo(denseOutput, dims, decodeOptions),
    nms_300_boxes: () => nms(b300.boxes, b300.scores, 0.45),
    nms_2000_boxes: () => nms(b2000.boxes, b2000.scores, 0.45),
    batched_nms_300_boxes_20_classes: () =>
      batchedNms(b300.boxes, b300.scores, b300.classIds, 0.45),
    batched_nms_2000_boxes_20_classes: () =>
      batchedNms(b2000.boxes, b2000.scores, b2000.classIds, 0.45),
    decode_yolo_seg_30_instances: () =>
      decodeYoloSeg(
        perAnchor,
        [1, 4 + NUM_CLASSES + NUM_MASK_COEFS, NUM_ANCHORS],
        prototypes,
        [1, NUM_MASK_COEFS, MASK_SIZE, MASK_SIZE],
        { ...decodeOptions, numClasses: NUM_CLASSES, inputWidth: 640, inputHeight: 640 },
      ),
  };
}

/**
 * Time a case and summarize the samples.
 *
 * Warm-up matters more here than in Python: V8 only optimizes a function after
 * it has run hot, so the first calls measure the interpreter.
 */
function measure(fn, reps, warmup) {
  for (let i = 0; i < warmup; i++) fn();
  const samples = [];
  for (let i = 0; i < reps; i++) {
    const start = performance.now();
    fn();
    samples.push(performance.now() - start);
  }
  samples.sort((a, b) => a - b);
  const round = (v) => Math.round(v * 10000) / 10000;
  return {
    median_ms: round(samples[Math.floor(samples.length / 2)]),
    min_ms: round(samples[0]),
    max_ms: round(samples[samples.length - 1]),
  };
}

function parseArgs(argv) {
  const args = { reps: 50, warmup: 10, json: null, compare: null, tolerance: DEFAULT_TOLERANCE };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === "--reps") args.reps = Number(value), i++;
    else if (key === "--warmup") args.warmup = Number(value), i++;
    else if (key === "--json") args.json = value, i++;
    else if (key === "--compare") args.compare = value, i++;
    else if (key === "--tolerance") args.tolerance = Number(value), i++;
  }
  return args;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const cases = buildCases();
  const results = {};
  for (const [name, fn] of Object.entries(cases)) results[name] = measure(fn, args.reps, args.warmup);
  const document = {
    environment: { node: process.version, platform: `${os.platform()}-${os.release()}`, arch: os.arch() },
    settings: { reps: args.reps, warmup: args.warmup },
    results,
  };

  if (args.json) {
    mkdirSync(dirname(args.json), { recursive: true });
    writeFileSync(args.json, JSON.stringify(document, null, 2) + "\n");
    console.log(`wrote ${args.json}`);
  }

  const names = Object.keys(results);
  const width = Math.max(...names.map((n) => n.length));
  if (!args.compare) {
    console.log(`${"case".padEnd(width)}  ${"median".padStart(10)}  ${"min".padStart(10)}  ${"max".padStart(10)}`);
    for (const name of names) {
      const r = results[name];
      console.log(
        `${name.padEnd(width)}  ${r.median_ms.toFixed(3).padStart(8)}ms  ${r.min_ms.toFixed(3).padStart(8)}ms  ${r.max_ms.toFixed(3).padStart(8)}ms`,
      );
    }
    return 0;
  }

  const baseline = JSON.parse(readFileSync(args.compare, "utf8"));
  if (JSON.stringify(baseline.environment) !== JSON.stringify(document.environment)) {
    console.log("! baseline was recorded on a different environment — numbers are not comparable");
  }
  let failed = 0;
  console.log(`${"case".padEnd(width)}  ${"baseline".padStart(10)}  ${"current".padStart(10)}  ${"delta".padStart(9)}`);
  for (const name of names) {
    const before = baseline.results[name]?.median_ms;
    const after = results[name].median_ms;
    if (before === undefined) {
      console.log(`${name.padEnd(width)}  ${"—".padStart(10)}  ${after.toFixed(3).padStart(8)}ms  ${"new".padStart(9)}`);
      continue;
    }
    const delta = before > 0 ? (after - before) / before : 0;
    const tooFast = Math.max(before, after) < NOISE_FLOOR_MS;
    const regressed = delta > args.tolerance && !tooFast;
    if (regressed) failed++;
    const flag = regressed ? "  REGRESSION" : tooFast ? "  below noise floor" : "";
    console.log(
      `${name.padEnd(width)}  ${before.toFixed(3).padStart(8)}ms  ${after.toFixed(3).padStart(8)}ms  ${((delta * 100).toFixed(1) + "%").padStart(9)}${flag}`,
    );
  }
  console.log(failed ? `\nfailed: ${failed} regression(s)` : "\nok: no case regressed");
  return failed ? 1 : 0;
}

process.exit(main());
