/**
 * Exceptions raised by the SDK.
 *
 * All exceptions inherit from {@link OrtVisionError}, so callers can catch
 * the base class to handle any SDK-originated failure uniformly.
 */

export class OrtVisionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** Raised when an ONNX model cannot be loaded into an inference session. */
export class ModelLoadError extends OrtVisionError {}

/** Raised when ONNX Runtime fails while executing a model. */
export class InferenceError extends OrtVisionError {}

/** Raised when a requested execution provider is not available. */
export class ProviderNotAvailableError extends OrtVisionError {}

/** Raised when an input image cannot be decoded into the canonical format. */
export class ImageLoadError extends OrtVisionError {}

/** Raised when class labels cannot be resolved from the supplied spec. */
export class LabelMapError extends OrtVisionError {}

/**
 * Raised when a model cannot be driven as a fused detect→classify pipeline.
 *
 * In the browser this means the file carries no `ovs.*` metadata — it is a
 * plain detector or classifier rather than something `ort_vision_sdk.compose`
 * produced — or the graph is missing an output the pipeline contract requires.
 * Building a pipeline is a Python-side build step; the browser only runs one.
 */
export class FusionError extends OrtVisionError {}
