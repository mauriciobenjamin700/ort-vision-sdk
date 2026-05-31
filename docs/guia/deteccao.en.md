# Detection

The `Detector` task supports anchor-free YOLO heads (v8, v9, v10, v11, v12,
v26). It letterboxes the image, runs the model, decodes the anchors, applies
NMS, and maps the boxes back to the original-image pixels.

## Building the detector

=== "Python"

    ```python
    from ort_vision_sdk import Detector

    det = Detector(
        "yolov8n.onnx",
        head="yolo",                # decoder family (default covers v8..v26)
        labels="coco",              # default — 80-class COCO preset
        input_size=(640, 640),      # default used for letterboxing
        conf_threshold=0.25,        # default minimum score
        iou_threshold=0.45,         # default NMS IoU
        max_detections=300,         # cap on detections per image
    )
    ```

=== "Web (browser)"

    ```typescript
    import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const det = await Detector.create("/models/yolov8n.onnx", {
      head: "yolo",                 // default
      labels: "coco",               // default
      inputSize: [640, 640],        // default
      confThreshold: 0.25,          // default
      iouThreshold: 0.45,           // default
    });
    ```

## Predicting

```python
result = det.predict("street.jpg")[0]
```

### The bulk `Boxes` view

The `boxes` view mirrors Ultralytics' `Boxes` interface:

```python
print(result.boxes.xyxy)    # (N, 4) absolute pixels [x1, y1, x2, y2]
print(result.boxes.xywh)    # (N, 4) [cx, cy, w, h]
print(result.boxes.xyxyn)   # (N, 4) normalized
print(result.boxes.xywhn)   # (N, 4) normalized [cx, cy, w, h]
print(result.boxes.cls)     # (N,) int64
print(result.boxes.conf)    # (N,) float64
print(result.boxes.data)    # (N, 6) [x1, y1, x2, y2, conf, cls]
```

On Web, `result.boxes` exposes the same attributes.

### Per-instance

```python
for d in result:
    print(d.name, d.conf, d.box.xyxy)
    # d.cropped_image: HWC uint8 RGB ndarray of the box crop
```

```typescript
for (const d of result) {
  console.log(d.className, d.confidence, d.bbox.asXyxy());
  // d.croppedImage: RGBImage of the box region
}
```

The Web `BoundingBox` exposes `asXyxy()` and `asXywh()`.

## Per-call overrides

You can override thresholds and filter classes on each `predict()`:

=== "Python"

    ```python
    result = det.predict(
        "img.jpg",
        conf_threshold=0.4,
        iou_threshold=0.5,
        classes=[0, 16],   # keep only these classes (e.g. person and dog)
    )[0]
    ```

=== "Web (browser)"

    ```typescript
    const result = (await det.predict("/img.jpg", {
      confThreshold: 0.4,
      iouThreshold: 0.5,
      classes: [0, 16],
    }))[0];
    ```

## Common patterns

### Filter by class

```python
people = [d for d in result if d.name == "person"]
```

### Save crops

```python
from PIL import Image
for i, d in enumerate(result):
    Image.fromarray(d.cropped_image).save(f"crop_{i}.png")
```

## See also

- [Quick start](../inicio-rapido.md)
- [Python API reference](../referencia/python.md)
- [Web API reference](../referencia/web.md)
