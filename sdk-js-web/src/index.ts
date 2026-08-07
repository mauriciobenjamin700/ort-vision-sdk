/**
 * @ort-vision-sdk/web — high-level TypeScript SDK for browser computer
 * vision inference with ONNX Runtime Web.
 */

export {
  BoundingBox,
  Mask,
  RGBImage,
  type ClassProbability,
  type ClassificationResult,
  type DetectionResult,
  type SegmentationResult,
} from "./types.js";

export {
  Boxes,
  ClassificationResults,
  DetectionResults,
  Masks,
  Probs,
  SegmentationResults,
} from "./results.js";

export {
  COCO_CLASSES,
  type LabelSpec,
  type ResolveLabelsOptions,
  defaultLabels,
  resolveLabels,
} from "./labels.js";

export {
  ImageLoadError,
  InferenceError,
  LabelMapError,
  ModelLoadError,
  OrtVisionError,
  ProviderNotAvailableError,
} from "./core/exceptions.js";

export {
  type ModelSource,
  type OrtSessionOptions,
  OrtSession,
} from "./core/session.js";
export {
  type DeclaredDim,
  type DeclaredShape,
  type ResolveInputSizeOptions,
  classificationNumClasses,
  declaredShapesFrom,
  detectionNumClasses,
  resolveInputSize,
  spatialInputSize,
} from "./core/graph.js";
export { modelNames, readModelMetadata } from "./core/metadata.js";
export { DEFAULT_PROVIDERS, resolveProviders } from "./core/providers.js";
export { type Speed, SpeedTimer } from "./core/timing.js";

export { type ImageInput, loadImage } from "./io/image.js";

export {
  type LetterboxResult,
  fromCv2,
  letterbox,
  normalize,
  resize,
  toCHW,
  toCv2,
  toFloat32,
  toFloat32Tensor,
  toTensor,
} from "./preprocess/image.js";

export {
  type FusedLetterboxResult,
  LetterboxPipeline,
  letterboxToTensorData,
  zeroTensorData,
} from "./preprocess/pipeline.js";

export {
  type TopKResult,
  softmax,
  topK,
} from "./postprocess/classification.js";

export {
  type DecodeYoloAnchorsOptions,
  type DecodeYoloOptions,
  type DecodedAnchors,
  type DecodedDetection,
  batchedNms,
  decodeYolo,
  decodeYoloAnchors,
  nms,
} from "./postprocess/detection.js";

export {
  type DecodeYoloSegOptions,
  type DecodedSegmentation,
  decodeYoloSeg,
} from "./postprocess/segmentation.js";

export { VisionTask } from "./tasks/base.js";
export {
  type ClassifierOptions,
  type ClassifierPredictOptions,
  Classifier,
} from "./tasks/classifier.js";
export {
  type DetectorHead,
  type DetectorOptions,
  type DetectorPredictOptions,
  Detector,
} from "./tasks/detector.js";
export {
  type SegmenterHead,
  type SegmenterOptions,
  type SegmenterPredictOptions,
  Segmenter,
} from "./tasks/segmenter.js";

export const VERSION: string = "0.5.1";
