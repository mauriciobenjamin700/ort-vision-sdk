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
export { DEFAULT_PROVIDERS, resolveProviders } from "./core/providers.js";

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
  type TopKResult,
  softmax,
  topK,
} from "./postprocess/classification.js";

export {
  type DecodeYoloAnchorsOptions,
  type DecodeYoloOptions,
  type DecodeYoloV8AnchorsOptions,
  type DecodeYoloV8Options,
  type DecodedAnchors,
  type DecodedDetection,
  batchedNms,
  decodeYolo,
  decodeYoloAnchors,
  decodeYoloV8,
  decodeYoloV8Anchors,
  nms,
} from "./postprocess/detection.js";

export {
  type DecodeYoloSegOptions,
  type DecodeYoloV8SegOptions,
  type DecodedSegmentation,
  decodeYoloSeg,
  decodeYoloV8Seg,
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

export const VERSION: string = "0.2.0";
