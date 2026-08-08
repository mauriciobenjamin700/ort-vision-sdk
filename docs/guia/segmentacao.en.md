# Segmentation

The `Segmenter` task supports YOLO-seg heads (v8-seg, v11-seg, v26-seg). It does
everything the detector does and, on top of that, decodes the mask prototypes
into per-instance binary masks.

## Building the segmenter

=== "Python"

    ```python
    from ort_vision_sdk import Segmenter

    seg = Segmenter(
        "yolov8n-seg.onnx",
        head="yolo-seg",            # decoder family (default)
        labels="coco",              # default — 80-class COCO preset
        input_size=(640, 640),      # default
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_detections=300,
        mask_threshold=0.5,         # soft → binary mask cutoff
    )
    ```

=== "Web (browser)"

    ```typescript
    import { Segmenter } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const seg = await Segmenter.create("/models/yolov8n-seg.onnx", {
      head: "yolo-seg",             // default
      labels: "coco",               // default
      inputSize: [640, 640],        // default
      confThreshold: 0.25,
      iouThreshold: 0.45,
      maskThreshold: 0.5,
    });
    ```

## Predicting

```python
result = seg.predict("street.jpg")[0]

# Same Boxes view as the detector …
print(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)

# … plus per-instance binary masks
for inst in result:
    print(inst.name, inst.conf, inst.box.xyxy)
    print(inst.mask.shape)            # (h, w) uint8 ∈ {0, 255}, cropped to the bbox
    print(inst.segmented_image.shape) # (h, w, 3) RGB with background zeroed
```

On Web:

```typescript
const result = (await seg.predict("/images/street.jpg"))[0];
for (const inst of result) {
  console.log(inst.className, inst.confidence, inst.bbox.asXyxy());
  console.log(inst.mask.width, inst.mask.height);  // cropped binary mask
  // inst.segmentedImage: RGBImage with the background zeroed out
}
```

## The `Masks` view

Beyond `boxes`, the segmentation envelope exposes the bulk `masks` view
(`masks.data`, `masks.xyxy`), mirroring Ultralytics' `Masks` interface.

Per instance, the mask is cropped to the bounding box:

- **Python:** `inst.mask` is an `(h, w)` uint8 ndarray with values in `{0, 255}`,
  and `inst.segmented_image` is the RGB crop with the background zeroed.
- **Web:** `inst.mask` is a `Mask` object (`data`/`width`/`height`, row-major
  layout), and `inst.segmentedImage` is an `RGBImage`.

## When finding nothing is an error

`Segmenter` takes the same `raise_on_empty` as `Detector`, with the same default
(`False`) and the same message — see
[When finding nothing is an error](deteccao.md#when-finding-nothing-is-an-error).

```python
seg = Segmenter("yolov8n-seg.onnx", conf_threshold=0.6, raise_on_empty=True)
seg.predict("img.jpg")   # nothing >= 0.6 -> NoDetectionsError
```

## See also

- [Detection guide](deteccao.md) — the same `Boxes` view and the same flow.
- [Python API reference](../referencia/python.md)
- [Web API reference](../referencia/web.md)
