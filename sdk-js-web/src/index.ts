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
  DetectClassifyResults,
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
  FusionError,
  ImageLoadError,
  InferenceError,
  LabelMapError,
  ModelLoadError,
  NoDetectionsError,
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
export { modelNames, parseNames, readModelMetadata } from "./core/metadata.js";
export { DEFAULT_PROVIDERS, resolveProviders } from "./core/providers.js";
export { type Speed, SpeedTimer } from "./core/timing.js";

export { type ImageInput, loadImage } from "./io/image.js";

export {
  FUSION_KIND_DETECT_CLASSIFY,
  INPUT_IMAGE,
  INPUT_PAD,
  INPUT_SCALE,
  INPUT_SOURCE,
  METADATA_PREFIX,
  OUTPUT_BOXES,
  OUTPUT_CLASSES,
  OUTPUT_NUM_DETECTIONS,
  OUTPUT_PROBS,
  OUTPUT_SCORES,
  type CropSource,
  type FusionSpec,
  readFusionSpec,
} from "./fusion.js";

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

export { VisionTask, requireDetections } from "./tasks/base.js";
export {
  type ClassifierOptions,
  type ClassifierPredictOptions,
  Classifier,
} from "./tasks/classifier.js";
export {
  type DetectClassifyOptions,
  type DetectClassifyPredictOptions,
  DetectClassify,
} from "./tasks/detectClassify.js";
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

export const VERSION: string = "0.6.0";
