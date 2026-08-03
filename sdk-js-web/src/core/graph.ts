/**
 * What the ONNX graph itself says about its inputs.
 *
 * The resolution a session must be fed at is a property of the exported model,
 * not of the configuration around it. Feeding a 640x640 tensor to a graph
 * exported at 224x224 makes ORT abort the run with
 * `Got invalid dimensions for input: images ... Got: 640 Expected: 224`, and the
 * caller has no way to see that coming from the outside — the number lives in
 * the file. So the SDK reads it from the graph and treats any configured size as
 * a fallback for when the graph leaves it open.
 */

import type * as ort from "onnxruntime-web";

/**
 * One declared dimension: a number when the graph pins it, `null` when the
 * dimension is symbolic (dynamic).
 */
export type DeclaredDim = number | null;

/** A declared input/output shape, dynamic axes appearing as `null`. */
export type DeclaredShape = readonly DeclaredDim[];

/**
 * Convert ORT value metadata into declared shapes.
 *
 * @param metadata Metadata as reported by `InferenceSession.inputMetadata`, or
 *   `undefined` on ORT builds that predate it (added in onnxruntime 1.21).
 * @returns One shape per value, in declaration order. Non-tensor values and
 *   builds without metadata yield empty shapes, which read as "nothing
 *   declared" everywhere downstream.
 */
export function declaredShapesFrom(
  metadata: readonly ort.InferenceSession.ValueMetadata[] | undefined,
): readonly DeclaredShape[] {
  if (metadata === undefined) return [];
  return metadata.map((value) =>
    value.isTensor
      ? value.shape.map((dim) => (typeof dim === "number" && Number.isInteger(dim) && dim > 0 ? dim : null))
      : [],
  );
}

/**
 * Read the spatial input size out of a declared NCHW shape.
 *
 * @param shape The declared shape of the model's image input.
 * @returns `[width, height]` in pixels, or `null` when the shape is not 4D or
 *   leaves either spatial axis dynamic — in which case the model accepts more
 *   than one resolution and there is nothing to correct.
 */
export function spatialInputSize(shape: DeclaredShape): readonly [number, number] | null {
  if (shape.length !== 4) return null;
  const height = shape[2];
  const width = shape[3];
  if (height === null || height === undefined || width === null || width === undefined) return null;
  return [width, height];
}

export interface ResolveInputSizeOptions {
  /** Declared shape of the model's image input, from {@link declaredShapesFrom}. */
  readonly graphShape?: DeclaredShape;
  /** Size the caller asked for, if any. */
  readonly requested?: readonly [number, number];
  /** Size to use when neither the graph nor the caller pins one. */
  readonly fallback: readonly [number, number];
}

/**
 * Decide the input size a task will preprocess to.
 *
 * Precedence is graph → caller → fallback. The graph wins over an explicit
 * `inputSize` because a static shape is not a preference, it is what ORT will
 * accept: honoring the caller there would only turn a fixable mismatch into a
 * failed run. A disagreement is a configuration bug in the caller, so it is
 * reported through `console.warn` instead of being swallowed.
 *
 * @param options Graph shape, requested size and per-task fallback.
 * @returns The `[width, height]` to preprocess to.
 */
export function resolveInputSize(options: ResolveInputSizeOptions): readonly [number, number] {
  const graph = options.graphShape === undefined ? null : spatialInputSize(options.graphShape);
  const requested = options.requested;
  if (graph === null) return requested ?? options.fallback;
  if (requested !== undefined && (requested[0] !== graph[0] || requested[1] !== graph[1])) {
    console.warn(
      `[ort-vision-sdk] The model declares a ${graph[0]}x${graph[1]} input; ` +
        `ignoring the requested ${requested[0]}x${requested[1]}, which ONNX Runtime would reject.`,
    );
  }
  return graph;
}
