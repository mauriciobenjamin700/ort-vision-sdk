# Segmentação

A tarefa `Segmenter` suporta cabeças YOLO-seg (v8-seg, v11-seg, v26-seg). Ela faz
tudo o que o detector faz e, além disso, decodifica os protótipos de máscara em
máscaras binárias por instância.

## Construindo o segmentador

=== "Python"

    ```python
    from ort_vision_sdk import Segmenter

    seg = Segmenter(
        "yolov8n-seg.onnx",
        head="yolo-seg",            # família de decoder (padrão)
        labels="coco",              # padrão — preset COCO de 80 classes
        input_size=(640, 640),      # padrão
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_detections=300,
        mask_threshold=0.5,         # corte soft → binário da máscara
    )
    ```

=== "Web (browser)"

    ```typescript
    import { Segmenter } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const seg = await Segmenter.create("/models/yolov8n-seg.onnx", {
      head: "yolo-seg",             // padrão
      labels: "coco",               // padrão
      inputSize: [640, 640],        // padrão
      confThreshold: 0.25,
      iouThreshold: 0.45,
      maskThreshold: 0.5,
    });
    ```

## Predizendo

```python
result = seg.predict("street.jpg")[0]

# Mesma visão Boxes do detector …
print(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)

# … mais máscaras binárias por instância
for inst in result:
    print(inst.name, inst.conf, inst.box.xyxy)
    print(inst.mask.shape)            # (h, w) uint8 ∈ {0, 255}, recortada na bbox
    print(inst.segmented_image.shape) # (h, w, 3) RGB com o fundo zerado
```

No Web:

```typescript
const result = (await seg.predict("/images/street.jpg"))[0];
for (const inst of result) {
  console.log(inst.className, inst.confidence, inst.bbox.asXyxy());
  console.log(inst.mask.width, inst.mask.height);  // máscara binária recortada
  // inst.segmentedImage: RGBImage com o fundo zerado
}
```

## A visão `Masks`

Além de `boxes`, o envelope de segmentação expõe a visão em massa `masks`
(`masks.data`, `masks.xyxy`), espelhando a interface `Masks` do Ultralytics.

Por instância, a máscara é recortada na bounding box:

- **Python:** `inst.mask` é um ndarray `(h, w)` uint8 com valores em `{0, 255}`,
  e `inst.segmented_image` é o recorte RGB com o fundo zerado.
- **Web:** `inst.mask` é um objeto `Mask` (`data`/`width`/`height`,
  layout row-major), e `inst.segmentedImage` é um `RGBImage`.

!!! check "Python e Web produzem a mesma máscara"
    Os dois SDKs seguem o mesmo algoritmo: combinam os protótipos, aplicam
    sigmoid, reamostram para a bounding box com bilinear de meio-pixel e
    binarizam no mesmo corte. Fixtures compartilhadas em `fixtures/parity/`
    verificam isso nos dois lados, comparando os bitmaps pixel a pixel — então
    se você roda o mesmo modelo no backend e no browser, as máscaras batem.

!!! warning "Máscaras geradas até a 0.6.0 diferem na borda"
    Até a 0.6.0 o lado Python reamostrava a máscara passando por `uint8`, o que
    colocava a entrada do teste `>= 0.5` numa grade de passos de `1/255` e
    deslocava pixels de borda sem motivo. Se você tem máscaras salvas de uma
    versão anterior, espere diferenças de alguns pixels na borda — as novas são
    as corretas (conferem 100% com uma referência em `float64`; as antigas,
    99,7%).

## Quando não segmentar nada é um erro

`Segmenter` aceita o mesmo `raise_on_empty` do `Detector`, com o mesmo default
(`False`) e a mesma mensagem — ver
[Quando não detectar nada é um erro](deteccao.md#quando-nao-detectar-nada-e-um-erro).

```python
seg = Segmenter("yolov8n-seg.onnx", conf_threshold=0.6, raise_on_empty=True)
seg.predict("img.jpg")   # nada >= 0.6 -> NoDetectionsError
```


## Veja também

- [Guia de detecção](deteccao.md) — a mesma visão `Boxes` e o mesmo fluxo.
- [Referência da API Python](../referencia/python.md)
- [Referência da API Web](../referencia/web.md)
