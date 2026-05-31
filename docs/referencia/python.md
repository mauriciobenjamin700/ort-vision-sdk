# Referência — API Python

Superfície pública do pacote `ort-vision-sdk` (tudo importável diretamente de
`ort_vision_sdk`).

## Tarefas

| Classe | Descrição |
| --- | --- |
| `Classifier` | Classificação de imagem (saída `(1, num_classes)`). |
| `Detector` | Detecção de objetos (cabeças YOLO anchor-free). |
| `Segmenter` | Segmentação de instância (cabeças YOLO-seg). |
| `VisionTask` | Classe base comum (não instancie diretamente). |
| `DetectorHead` | Tipo das famílias de decoder de detecção (ex.: `"yolo"`). |
| `SegmenterHead` | Tipo das famílias de decoder de segmentação (ex.: `"yolo-seg"`). |

Cada tarefa expõe três variantes de inferência com a **mesma assinatura**:
`predict()`, `async_predict()` (`asyncio.to_thread`) e `ort_async_predict()`
(`InferenceSession.run_async`). Todas retornam `list[Results]` de comprimento 1
por imagem.

### Construtores (resumo)

```python
Classifier(model_path, *, labels=None, providers=None, session_options=None,
           input_size=(224, 224), mean=..., std=..., apply_softmax=True)

Detector(model_path, *, head="yolo", labels="coco", providers=None,
         session_options=None, input_size=(640, 640), conf_threshold=0.25,
         iou_threshold=0.45, max_detections=300)

Segmenter(model_path, *, head="yolo-seg", labels="coco", providers=None,
          session_options=None, input_size=(640, 640), conf_threshold=0.25,
          iou_threshold=0.45, max_detections=300, mask_threshold=0.5)
```

`Detector.predict()` e `Segmenter.predict()` aceitam overrides por chamada:
`conf_threshold`, `iou_threshold`, `classes`.

## Envelopes de resultado

| Envelope | Visão em massa | Iterar produz | Campos notáveis |
| --- | --- | --- | --- |
| `ClassificationResults` | `probs` | n/a (resultado único) | `cls`, `conf`, `name`, `probabilities` |
| `DetectionResults` | `boxes` | `DetectionResult` | `cls`, `conf`, `box.xyxy`, `cropped_image` |
| `SegmentationResults` | `boxes`, `masks` | `SegmentationResult` | `cls`, `conf`, `box.xyxy`, `mask`, `segmented_image` |

Todo envelope expõe também `names`, `orig_img`, `orig_shape`, `path` e um
`speed` opcional (timings).

## Visões em massa (estilo Ultralytics)

| Classe | Atributos |
| --- | --- |
| `Boxes` | `xyxy`, `xywh`, `xyxyn`, `xywhn`, `cls`, `conf`, `data` |
| `Probs` | `top1`, `top5`, `top1conf`, `top5conf`, `data` |
| `Masks` | `data`, `xyxy` |

## Tipos por instância

| Tipo | Campos canônicos | Aliases Ultralytics |
| --- | --- | --- |
| `DetectionResult` | `class_id`, `class_name`, `confidence`, `bbox`, `cropped_image` | `cls`, `name`, `conf`, `box` |
| `SegmentationResult` | + `mask`, `segmented_image` | `cls`, `name`, `conf`, `box` |
| `ClassificationResult` | `class_id`, `class_name`, `confidence` | `cls`, `name`, `conf` |
| `ClassProbability` | `class_id`, `class_name`, `probability` | `cls`, `name` |
| `BoundingBox` | `x1`, `y1`, `x2`, `y2` + `xyxy` | — |

## Imagens e rótulos

| Símbolo | Descrição |
| --- | --- |
| `load_image(image)` | Carrega qualquer entrada suportada para um `ndarray` HWC uint8 RGB. |
| `ImageInput` | Tipo de união das entradas aceitas por `predict()`. |
| `ImageArray` | Alias para o `ndarray` HWC uint8 RGB. |
| `resolve_labels(spec, ...)` | Resolve uma `LabelSpec` para `dict[int, str]`. |
| `LabelSpec` | Tipo de união aceito por `labels=` (preset, lista, dict, path, None). |
| `COCO_CLASSES` | Tupla com as 80 classes do preset COCO. |

!!! note "Fonte da verdade"
    As assinaturas completas, com tipos e docstrings, vivem no código-fonte em
    [`sdk-python/src/ort_vision_sdk/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-python/src/ort_vision_sdk).
    Esta página resume a superfície pública exportada em `__init__.py`.
