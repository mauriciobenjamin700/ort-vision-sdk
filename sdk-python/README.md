# ort-vision-sdk

High-level Python SDK for computer vision inference on top of [ONNX Runtime](https://onnxruntime.ai/).

Wraps the low-level `InferenceSession` API with task-oriented classes (`Classifier`, `Detector`, ...) that handle preprocessing, execution provider selection, and postprocessing — so you go from an image to a typed result in one call.

## Installation

```bash
pip install ort-vision-sdk          # CPU only
pip install ort-vision-sdk[gpu]     # CUDA
pip install ort-vision-sdk[opencv]  # adds cv2 image backend
```

## Quick start

```python
from ort_vision_sdk import Classifier

clf = Classifier("resnet50.onnx", labels="imagenet")
result = clf.predict("dog.jpg")

print(result.class_name, result.confidence)
print(result.probabilities[:5])  # top-5 ClassProbability tuples
```

```python
from ort_vision_sdk import Detector

det = Detector("yolov8n.onnx", labels="coco")
detections = det.predict("street.jpg")

for d in detections:
    print(d.class_name, d.confidence, d.bbox.as_xyxy())
    # d.cropped_image is a np.ndarray (HWC, RGB, uint8)
```

## Status

Alpha — API may change. See [`pyproject.toml`](pyproject.toml) for supported Python and dependency versions.
