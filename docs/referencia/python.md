# Referência — API Python

Superfície pública do pacote `ort-vision-sdk` (tudo importável diretamente de
`ort_vision_sdk`).

## Tarefas

| Classe | Descrição |
| --- | --- |
| `Classifier` | Classificação de imagem (saída `(1, num_classes)`). |
| `Detector` | Detecção de objetos (cabeças YOLO anchor-free). |
| `Segmenter` | Segmentação de instância (cabeças YOLO-seg). |
| `DetectClassify` | Pipeline fundido detector → classificador, num único `.onnx`. |
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
           backend=None, input_size=None, mean=..., std=..., apply_softmax=True)

Detector(model_path, *, head="yolo", labels=None, providers=None,
         session_options=None, backend=None, input_size=None,
         conf_threshold=0.25, iou_threshold=0.45, max_detections=300,
         raise_on_empty=False)

Segmenter(model_path, *, head="yolo-seg", labels=None, providers=None,
          session_options=None, backend=None, input_size=None,
          conf_threshold=0.25, iou_threshold=0.45, max_detections=300,
          mask_threshold=0.5, raise_on_empty=False)

DetectClassify(model_path, *, labels=None, classifier_labels=None,
               raise_on_empty=False, providers=None, session_options=None,
               backend=None)
```

`DetectClassify` tem só esses parâmetros porque todo o resto — resolução,
tamanho do recorte, limiares, softmax, nomes de classe dos dois estágios — foi
gravado no arquivo na hora da fusão. Ver
[Pipelines fundidos](../guia/pipeline.md).

Os três construtores aceitam `backend=` (v0.4.0): injeta um `InferenceBackend`
para rodar a inferência fora do ONNX Runtime in-process (navegador, Android).
Quando fornecido, `model_path`/`providers`/`session_options` são ignorados. Veja
o [guia de backends](../guia/backends.md). (Adicionei `backend=None` às
assinaturas de `Classifier`/`Detector` acima também.)

`input_size=None` e `labels=None` (v0.6.0) significam "pergunte ao modelo":
a resolução vem do shape declarado pelo grafo e os nomes vêm dos `names` nos
metadados, com os defaults antigos (224/640, preset COCO) como fallback. Ver
[O modelo manda](../guia/modelo.md).

`Detector.predict()` e `Segmenter.predict()` aceitam overrides por chamada:
`conf_threshold`, `iou_threshold`, `classes`.

## Backends de inferência

| Símbolo | Descrição |
| --- | --- |
| `InferenceBackend` | Protocolo do motor de inferência — metadata (`input_name`/`input_shape`/`output_names`/`output_shapes`) + `run`/`async_run`/`ort_async_run`. |
| `MetadataBackend` | Protocolo de capacidade: backends que leem o mapa de metadados do modelo. Separado porque um bridge nativo pode não conseguir. |
| `read_metadata(backend)` | Lê o mapa de metadados de um backend, devolvendo `{}` quando ele não oferece a capacidade. |
| `OrtSession` | Backend padrão (ONNX Runtime in-process); satisfaz os dois protocolos. |
| `OrtSession.metadata` | Mapa de metadados customizados do modelo (`names`, `task`, `imgsz`, ...). |
| `OrtSession.input_shape` | Shape declarado da primeira entrada (eixos dinâmicos como string). |

## Envelopes de resultado

| Envelope | Visão em massa | Iterar produz | Campos notáveis |
| --- | --- | --- | --- |
| `ClassificationResults` | `probs` | n/a (resultado único) | `cls`, `conf`, `name`, `probabilities` |
| `DetectionResults` | `boxes` | `DetectionResult` | `cls`, `conf`, `box.xyxy`, `cropped_image` |
| `DetectClassifyResults` | `boxes` | `DetectionResult` | + `classification`, e `classifier_names` no envelope |
| `SegmentationResults` | `boxes`, `masks` | `SegmentationResult` | `cls`, `conf`, `box.xyxy`, `mask`, `segmented_image` |

Todo envelope expõe também `names`, `orig_img`, `orig_shape`, `path` e
`speed` — um `dict[str, float]` com `load`, `preprocess`, `inference` e
`postprocess` em milissegundos, preenchido por todo `predict()`. Ver
[Custo da inferência](../guia/velocidade.md).

## Visões em massa (estilo Ultralytics)

| Classe | Atributos |
| --- | --- |
| `Boxes` | `xyxy`, `xywh`, `xyxyn`, `xywhn`, `cls`, `conf`, `data` |
| `Probs` | `top1`, `top5`, `top1conf`, `top5conf`, `data` |
| `Masks` | `data`, `xyxy` |

## Tipos por instância

| Tipo | Campos canônicos | Aliases Ultralytics |
| --- | --- | --- |
| `DetectionResult` | `class_id`, `class_name`, `confidence`, `bbox`, `cropped_image`, `classification` | `cls`, `name`, `conf`, `box` |
| `SegmentationResult` | + `mask`, `segmented_image` | `cls`, `name`, `conf`, `box` |
| `ClassificationResult` | `class_id`, `class_name`, `confidence` | `cls`, `name`, `conf` |
| `ClassProbability` | `class_id`, `class_name`, `probability` | `cls`, `name` |
| `BoundingBox` | `x1`, `y1`, `x2`, `y2` + `xyxy` | — |

## Resultado vazio

`Detector`, `Segmenter` e `DetectClassify` aceitam `raise_on_empty` no construtor
e como override em cada `predict()`. Default `False`: não achar nada devolve um
envelope vazio, não um erro. Com `True`, levanta
`NoDetectionsError` — ver
[Quando não detectar nada é um erro](../guia/deteccao.md#quando-nao-detectar-nada-e-um-erro).

| Símbolo | Descrição |
| --- | --- |
| `raise_on_empty` | Argumento de construtor e de `predict()`; o valor por chamada vence o do construtor. |
| `NoDetectionsError` | Levantado quando nada sobra e o flag está ativo. Exportado em `ort_vision_sdk.core`. |

## Compondo pipelines (extra `[compose]`)

| Símbolo | Descrição |
| --- | --- |
| `compose.fuse_detect_classify(...)` | Funde um detector YOLO e um classificador num único `.onnx`, e valida o resultado rodando-o. |
| `compose.build_bridge(...)` | Monta só o subgrafo-ponte (NMS → RoiAlign → normalização). Útil para inspeção. |
| `compose.MIN_OPSET` | Opset mínimo que a ponte exige (16, por causa do `RoiAlign`). |
| `FusionError` | Erro levantado quando dois modelos não podem ser fundidos, ou o arquivo carregado não é um pipeline. |

Este módulo é o único que importa `onnx`, e só é instalado com
`pip install "ort-vision-sdk[compose]"`. Rodar o modelo fundido não precisa
dele. Ver [Pipelines fundidos](../guia/pipeline.md).

## Imagens e rótulos

| Símbolo | Descrição |
| --- | --- |
| `load_image(image)` | Carrega qualquer entrada suportada para um `ndarray` HWC uint8 RGB. |
| `ImageInput` | Tipo de união das entradas aceitas por `predict()`. |
| `ImageArray` | Alias para o `ndarray` HWC uint8 RGB. |
| `resolve_labels(spec, ...)` | Resolve uma `LabelSpec` para `dict[int, str]`. |
| `LabelSpec` | Tipo de união aceito por `labels=` (preset, lista, dict, path, None). |
| `COCO_CLASSES` | Tupla com as 80 classes do preset COCO. |

## O que o modelo declara

| Símbolo | Descrição |
| --- | --- |
| `spatial_input_size(shape)` | Extrai `(largura, altura)` de um shape NCHW estático; `None` quando os eixos são dinâmicos. |
| `resolve_input_size(...)` | Aplica a precedência grafo → chamador → fallback, avisando quando o chamador contradiz um grafo estático. |
| `model_names(metadata)` | Interpreta o `names` do Ultralytics (`repr` de `dict[int, str]`) via `ast.literal_eval`; `None` quando ausente ou inutilizável. |
| `parse_names(raw)` | O mesmo parser, sobre uma string qualquer — usado pelos dois mapas de classe de um pipeline fundido. |
| `FusionSpec` | O que um pipeline fundido declara sobre si mesmo; `FusionSpec.from_metadata(...)` o lê de volta. |
| `CropSource` | `"detector_input"` ou `"original"` — de onde a ponte recorta as caixas. |
| `task.input_size` | Resolução em que a tarefa realmente pré-processa. |

!!! note "Fonte da verdade"
    As assinaturas completas, com tipos e docstrings, vivem no código-fonte em
    [`sdk-python/src/ort_vision_sdk/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-python/src/ort_vision_sdk).
    Esta página resume a superfície pública exportada em `__init__.py`.
