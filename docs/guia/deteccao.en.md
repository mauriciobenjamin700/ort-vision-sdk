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

## When finding nothing is an error

By default, a `predict()` that finds nothing returns an **empty** envelope, not
an exception. That is deliberate: the model looked and found nothing is a
successful inference — a photo of an empty field is a valid photo.

But the opposite case exists: a step whose **precondition** is that something is
there, where carrying on with zero rows is worse than stopping. For that, turn
on `raise_on_empty`:

=== "Python"

    ```python
    from ort_vision_sdk import Detector
    from ort_vision_sdk.core import NoDetectionsError

    det = Detector("yolov8n.onnx", conf_threshold=0.7, raise_on_empty=True)

    try:
        result = det.predict("img.jpg")[0]
    except NoDetectionsError as error:
        print(error)
        # No detections in img.jpg: nothing cleared conf_threshold=0.7.
    ```

=== "Web (browser)"

    ```typescript
    import { Detector, NoDetectionsError } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const det = await Detector.create("/models/yolov8n.onnx", {
      confThreshold: 0.7,
      raiseOnEmpty: true,
    });

    try {
      const result = (await det.predict("/img.jpg"))[0];
    } catch (error) {
      if (error instanceof NoDetectionsError) console.log(error.message);
    }
    ```

!!! tip "\"Nothing detected\" and \"nothing confident enough\" are the same case"
    What decides whether something **counts** as a detection is
    `conf_threshold`. So there is no separate threshold just for the exception:
    raise `conf_threshold` and the error covers the stricter bar. The message
    always names the threshold that applied to the call — without it you cannot
    tell a blank image from a threshold set too high.

The message also names the image (when the input was a path) and the class
filter, whenever one of them narrowed the search:

```text
No detections in flock.jpg among classes [0, 16]: nothing cleared conf_threshold=0.25.
```

And you can flip it per call, in either direction:

```python
det = Detector("yolov8n.onnx", raise_on_empty=True)

det.predict("img.jpg", raise_on_empty=False)   # this call returns [] in peace
det.predict("img.jpg", conf_threshold=0.9)     # raise the bar; the error follows
```

The flag exists identically on `Detector`, `Segmenter` and `DetectClassify`,
with the same default (`False`) and the same message.

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
