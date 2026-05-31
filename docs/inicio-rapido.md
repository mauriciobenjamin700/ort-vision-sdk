# Início rápido

Este guia mostra o caminho mais curto até um resultado tipado nas duas
plataformas. Cada `predict()` retorna uma **lista de envelopes de comprimento 1
por imagem** — por isso o padrão `[0]`. O envelope é iterável: percorrê-lo
produz as instâncias detectadas.

## Python

### Classificação

```python
from ort_vision_sdk import Classifier

clf = Classifier(
    "resnet50.onnx",
    labels="imagenet_labels.txt",  # uma classe por linha, ou uma lista/dict
    input_size=(224, 224),         # padrão
    apply_softmax=True,            # False se o modelo já emite probabilidades
)

r = clf.predict("dog.jpg")[0]
print(r.cls, r.conf, r.name)       # top-1 — estilo Ultralytics
print(r.probs.top5)                # array com os índices top-5
print(r.probs.top5conf)            # probabilidades correspondentes
```

### Detecção

```python
from ort_vision_sdk import Detector

det = Detector(
    "yolov8n.onnx",
    labels="coco",                 # padrão — preset COCO de 80 classes
    conf_threshold=0.25,
    iou_threshold=0.45,
)

result = det.predict("street.jpg")[0]

# Visão em massa — interface Boxes do Ultralytics
print(result.boxes.xyxy.shape)     # (N, 4) em pixels absolutos
print(result.boxes.cls, result.boxes.conf)

# Por instância (dataclasses)
for d in result:
    print(d.name, d.conf, d.box.xyxy)
    # d.cropped_image é um ndarray HWC uint8 RGB do recorte da caixa
```

### Segmentação

```python
from ort_vision_sdk import Segmenter

seg = Segmenter("yolov8n-seg.onnx", labels="coco", mask_threshold=0.5)
result = seg.predict("street.jpg")[0]

for inst in result:
    print(inst.name, inst.conf, inst.box.xyxy)
    print(inst.mask.shape)          # (h, w) uint8 ∈ {0, 255}, recortada na bbox
    print(inst.segmented_image.shape)  # (h, w, 3) RGB com fundo zerado
```

### Inferência assíncrona

Cada classe expõe `async_predict()` (off-load via `asyncio.to_thread`, caminho
async padrão) e `ort_async_predict()` (`InferenceSession.run_async`, para alta
concorrência). Veja o [guia Python](guia/python.md) para a diferença.

```python
result = (await det.async_predict("street.jpg"))[0]
```

## Web (browser)

A API espelha a do Python. As tarefas são criadas com `await Task.create(...)`
(carrega o modelo de forma assíncrona) e `predict()` é sempre `async`.

### Classificação

```typescript
import { Classifier } from "@mauriciobenjamin700/ort-vision-sdk-web";

const clf = await Classifier.create("/models/resnet50.onnx", {
  labels: ["tench", "goldfish", /* ... 1000 rótulos ImageNet */],
});

const r = (await clf.predict("/images/dog.jpg", { topK: 5 }))[0];
console.log(r.cls, r.conf, r.name);
console.log(r.probs.top5, r.probs.top5conf);
```

### Detecção

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

// labels assume "coco" (80 classes) por padrão
const det = await Detector.create("/models/yolov8n.onnx");

const result = (await det.predict("/images/street.jpg", { confThreshold: 0.4 }))[0];
for (const d of result) {
  console.log(d.className, d.confidence, d.bbox.asXyxy());
  // d.croppedImage é um RGBImage só da região da caixa
}
```

### Segmentação

```typescript
import { Segmenter } from "@mauriciobenjamin700/ort-vision-sdk-web";

const seg = await Segmenter.create("/models/yolov8n-seg.onnx", { maskThreshold: 0.5 });
const result = (await seg.predict("/images/street.jpg"))[0];
for (const inst of result) {
  console.log(inst.className, inst.confidence, inst.bbox.asXyxy());
  console.log(inst.mask.width, inst.mask.height);
}
```

## Problemas comuns

| Sintoma | Causa / solução |
|---|---|
| `ModuleNotFoundError: ort_vision_sdk` | Pacote não instalado no ambiente. Rode `pip install ort-vision-sdk`. Veja [Instalação](instalacao.md). |
| Erro de import do `onnxruntime-web` no navegador | É *peer dependency* — instale-o ao lado do SDK e sirva os `.wasm` correspondentes. |
| Arquivo de modelo não encontrado / falha ao carregar | Confira o caminho do `.onnx`. Exporte um com `yolo export model=yolov8n.pt format=onnx` (veja [Instalação](instalacao.md)). |
| `predict(...)` parece devolver "uma lista" estranha | É por design: cada `predict` devolve uma **lista de um envelope por imagem** — pegue `[0]`. |
| Rótulos saem como `class_0`, `class_1`, … | O modelo não trouxe nomes e nenhum `labels` foi passado. Passe `labels="coco"`, uma lista, um dict ou um arquivo — veja [Guia Python](guia/python.md#rotulos). |
| Inferência trava o servidor `async` | Não chame `predict` síncrono num handler `async`; use `async_predict`. Veja [Guia Python](guia/python.md#inferencia-assincrona). |

## Recapitulando

- Cada tarefa é uma classe (`Classifier`, `Detector`, `Segmenter`); `predict`
  devolve uma **lista de um envelope** por imagem — use `[0]`.
- Itere o envelope para os itens, ou use as views em bloco (`.boxes`, `.probs`).
- A API é a mesma em Python e no navegador; só muda `snake_case` ↔ `camelCase`.

## Próximos passos

- Guias por tarefa: [classificação](guia/classificacao.md),
  [detecção](guia/deteccao.md), [segmentação](guia/segmentacao.md).
- Diferenças por plataforma: [Python](guia/python.md) e [Web](guia/web.md).
- Referência completa: [API Python](referencia/python.md) e
  [API Web](referencia/web.md).
