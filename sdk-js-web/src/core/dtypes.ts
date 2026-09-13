/**
 * What element type the graph declares, and what typed array that means.
 *
 * A model exported with `half=True` declares its input as `float16`, and ONNX
 * Runtime refuses a float32 feed against it:
 *
 * ```text
 * Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(float16))
 * ```
 *
 * The session opens without complaint, so the failure lands on the caller's
 * first `predict()` rather than on `create()`. Preprocessing stays in
 * `Float32Array` because that is where the normalization arithmetic belongs —
 * `(value / 255 - mean) / std` in half precision loses exactly the small
 * differences normalization exists to preserve — and the conversion happens
 * once, at the feed boundary.
 *
 * Outputs travel the other way. Float16 carries 11 bits of mantissa, so at
 * pixel scale its neighbouring values are 0.5 apart around coordinate 640 and
 * 1.0 apart around 1280: decoding boxes in that type quantises every coordinate
 * before NMS and the scale-back to original-image pixels ever run. {@link
 * asFloat32Array} widens a half-precision output before the decoders touch it.
 *
 * `Float16Array` is what ORT requires for a half tensor — a `Uint16Array` of
 * the same bits is rejected — and it is not in every browser yet, so
 * {@link hasFloat16Array} exists to refuse at session creation instead of on
 * the first frame.
 */

/** ONNX `TensorProto.DataType` values this SDK can name. */
const ELEM_TYPE_TO_TENSOR_TYPE: Readonly<Record<number, string>> = {
  1: "float32",
  2: "uint8",
  3: "int8",
  4: "uint16",
  5: "int16",
  6: "int32",
  7: "int64",
  9: "bool",
  10: "float16",
  11: "float64",
  12: "uint32",
  13: "uint64",
};

/** The element type every task preprocesses to, and the fallback when unknown. */
export const DEFAULT_TENSOR_TYPE = "float32";

/**
 * The structural surface of `Float16Array` this module needs.
 *
 * Declared here rather than pulled from `lib.es2025`: the package targets
 * ES2022 so it builds for consumers on older TypeScript, and the only thing
 * needed is "an array-like of numbers ORT recognises as half precision".
 */
export interface Float16ArrayLike {
  readonly length: number;
  readonly buffer: ArrayBufferLike;
  [index: number]: number;
}

interface Float16ArrayConstructor {
  new (input: ArrayLike<number> | number): Float16ArrayLike;
}

const FLOAT16_ARRAY = (globalThis as { Float16Array?: Float16ArrayConstructor }).Float16Array;

/**
 * Whether this environment can build a half-precision tensor at all.
 *
 * @returns `true` when the runtime exposes `Float16Array`. ORT rejects any
 *   other typed array for a `float16` tensor, so a `false` here means half
 *   precision is unreachable in this browser no matter what the model declares.
 */
export function hasFloat16Array(): boolean {
  return FLOAT16_ARRAY !== undefined;
}

/**
 * Name the tensor type an ONNX element type maps to.
 *
 * @param elemType A `TensorProto.DataType` value read off the graph.
 * @returns The ORT tensor type name, or `"float32"` for anything unrecognised —
 *   which is what every task produces anyway, so an unknown type surfaces as
 *   ORT's own type error against the real graph rather than as a lookup miss.
 */
export function tensorTypeFor(elemType: number | undefined): string {
  if (elemType === undefined) return DEFAULT_TENSOR_TYPE;
  return ELEM_TYPE_TO_TENSOR_TYPE[elemType] ?? DEFAULT_TENSOR_TYPE;
}

/**
 * Convert a preprocessed float32 buffer to the type a feed must carry.
 *
 * @param data The preprocessed planar buffer.
 * @param tensorType The tensor type the input declares, from
 *   {@link tensorTypeFor}.
 * @returns `data` itself for a float32 input, or a `Float16Array` holding the
 *   same values for a half-precision one.
 * @throws {RangeError} If half precision is asked for and the runtime has no
 *   `Float16Array`. Callers convert this into a `ModelLoadError` at session
 *   creation, so it never reaches a `predict()`.
 */
export function toFeedData(data: Float32Array, tensorType: string): Float32Array | Float16ArrayLike {
  if (tensorType !== "float16") return data;
  if (FLOAT16_ARRAY === undefined) {
    throw new RangeError(
      "This model declares a float16 input, but this environment has no Float16Array.",
    );
  }
  return new FLOAT16_ARRAY(data);
}

/**
 * Widen a model output to `Float32Array`, without copying when it already is.
 *
 * @param data A tensor's `data`, as ORT returned it.
 * @returns The same values as a `Float32Array`. A float32 output is passed
 *   through untouched; anything else is copied once.
 */
export function asFloat32Array(data: unknown): Float32Array {
  if (data instanceof Float32Array) return data;
  return Float32Array.from(data as ArrayLike<number>);
}
