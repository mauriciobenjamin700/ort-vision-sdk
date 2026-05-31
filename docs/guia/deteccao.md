# Detecção

A tarefa `Detector` suporta cabeças YOLO anchor-free (v8, v9, v10, v11, v12,
v26). Ela faz letterbox da imagem, executa o modelo, decodifica as âncoras,
aplica NMS e mapeia as caixas de volta para os pixels da imagem original.

## Construindo o detector

=== "Python"

    ```python
    from ort_vision_sdk import Detector

    det = Detector(
        "yolov8n.onnx",
        head="yolo",                # família de decoder (padrão cobre v8..v26)
        labels="coco",              # padrão — preset COCO de 80 classes
        input_size=(640, 640),      # padrão usado no letterbox
        conf_threshold=0.25,        # score mínimo por padrão
        iou_threshold=0.45,         # IoU do NMS por padrão
        max_detections=300,         # teto de detecções por imagem
    )
    ```

=== "Web (browser)"

    ```typescript
    import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const det = await Detector.create("/models/yolov8n.onnx", {
      head: "yolo",                 // padrão
      labels: "coco",               // padrão
      inputSize: [640, 640],        // padrão
      confThreshold: 0.25,          // padrão
      iouThreshold: 0.45,           // padrão
    });
    ```

## Predizendo

```python
result = det.predict("street.jpg")[0]
```

### Visão em massa `Boxes`

A visão `boxes` espelha a interface `Boxes` do Ultralytics:

```python
print(result.boxes.xyxy)    # (N, 4) pixels absolutos [x1, y1, x2, y2]
print(result.boxes.xywh)    # (N, 4) [cx, cy, w, h]
print(result.boxes.xyxyn)   # (N, 4) normalizado
print(result.boxes.xywhn)   # (N, 4) normalizado [cx, cy, w, h]
print(result.boxes.cls)     # (N,) int64
print(result.boxes.conf)    # (N,) float64
print(result.boxes.data)    # (N, 6) [x1, y1, x2, y2, conf, cls]
```

No Web, `result.boxes` expõe os mesmos atributos.

### Por instância

```python
for d in result:
    print(d.name, d.conf, d.box.xyxy)
    # d.cropped_image: ndarray HWC uint8 RGB do recorte da caixa
```

```typescript
for (const d of result) {
  console.log(d.className, d.confidence, d.bbox.asXyxy());
  // d.croppedImage: RGBImage da região da caixa
}
```

A `BoundingBox` no Web expõe `asXyxy()` e `asXywh()`.

## Overrides por chamada

Você pode sobrescrever os thresholds e filtrar classes em cada `predict()`:

=== "Python"

    ```python
    result = det.predict(
        "img.jpg",
        conf_threshold=0.4,
        iou_threshold=0.5,
        classes=[0, 16],   # mantém só essas classes (ex.: pessoa e cão)
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

## Padrões comuns

### Filtrar por classe

```python
people = [d for d in result if d.name == "person"]
```

### Salvar recortes

```python
from PIL import Image
for i, d in enumerate(result):
    Image.fromarray(d.cropped_image).save(f"crop_{i}.png")
```

## Veja também

- [Início rápido](../inicio-rapido.md)
- [Referência da API Python](../referencia/python.md)
- [Referência da API Web](../referencia/web.md)
