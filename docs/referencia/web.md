# Referência — API Web

Superfície pública do pacote `@mauriciobenjamin700/ort-vision-sdk-web` (tudo
exportado de `@mauriciobenjamin700/ort-vision-sdk-web`).

## Tarefas

| Classe | Criação | Descrição |
| --- | --- | --- |
| `Classifier` | `await Classifier.create(model, options)` | Classificação de imagem. |
| `Detector` | `await Detector.create(model, options)` | Detecção de objetos (YOLO). |
| `Segmenter` | `await Segmenter.create(model, options)` | Segmentação de instância (YOLO-seg). |
| `VisionTask` | — | Classe base comum. |

`predict()` é sempre `async` e retorna `Promise<...Results[]>` de comprimento 1
por imagem. Cada tarefa expõe um alias `run()`.

### Tipos de opções

| Tipo | Para |
| --- | --- |
| `ClassifierOptions` / `ClassifierPredictOptions` | construção / `predict` do `Classifier` (`labels`, `numClasses`, `inputSize`, `applySoftmax`, `providers`; `topK` no predict) |
| `DetectorOptions` / `DetectorPredictOptions` | `Detector` (`head`, `labels`, `inputSize`, `confThreshold`, `iouThreshold`; overrides + `classes` no predict) |
| `SegmenterOptions` / `SegmenterPredictOptions` | `Segmenter` (+ `maskThreshold`) |
| `DetectorHead` (`"yolo"`) / `SegmenterHead` (`"yolo-seg"`) | famílias de decoder |

## Resultados

| Envelope | Visão em massa | Iterar produz |
| --- | --- | --- |
| `ClassificationResults` | `probs` | n/a (resultado único) |
| `DetectionResults` | `boxes` | `DetectionResult` |
| `SegmentationResults` | `boxes`, `masks` | `SegmentationResult` |

Todo envelope expõe `names`, `origImg`, `origShape`, `path` e `speed` — um
objeto `Speed` com `load`, `preprocess`, `inference` e `postprocess` em
milissegundos, preenchido por todo `predict()`. Ver
[Custo da inferência](../guia/velocidade.md).

Visões em massa: `Boxes`, `Probs`, `Masks` (mesmos atributos do Python).

Tipos/classes por instância: `DetectionResult`, `SegmentationResult`,
`ClassificationResult`, `ClassProbability` (com `classId`/`className`/
`confidence` e os aliases `cls`/`name`/`conf`/`box`), além de `BoundingBox`
(`asXyxy()`, `asXywh()`), `Mask` (`data`/`width`/`height`) e `RGBImage`.

## Imagens, rótulos e providers

| Símbolo | Descrição |
| --- | --- |
| `loadImage(image)` | Carrega qualquer entrada suportada para um `RGBImage`. |
| `ImageInput` | Tipo de união das entradas aceitas por `predict()`. |
| `resolveLabels(spec, options)` | Resolve uma `LabelSpec` para o mapeamento de classes. |
| `LabelSpec` / `ResolveLabelsOptions` | Tipos da resolução de rótulos. |
| `COCO_CLASSES` | As 80 classes do preset COCO. |
| `DEFAULT_PROVIDERS` | `["webgpu", "wasm"]`. |
| `resolveProviders(...)` | Resolve a lista de providers para nomes do ORT-Web. |
| `OrtSession` / `OrtSessionOptions` / `ModelSource` | Sessão de baixo nível. |

## Erros

Hierarquia de exceções exportada: `OrtVisionError` (base), `ImageLoadError`,
`InferenceError`, `LabelMapError`, `ModelLoadError`,
`ProviderNotAvailableError`.

## Utilitários de pré/pós-processamento

O pacote também exporta helpers de baixo nível para quem constrói o próprio
pipeline: `letterbox`, `resize`, `normalize`, `toCHW`, `toTensor`,
`toFloat32`/`toFloat32Tensor`, `fromCv2`/`toCv2`, `softmax`, `topK`, `nms`,
`batchedNms`, `decodeYolo`/`decodeYoloV8`, `decodeYoloAnchors`/
`decodeYoloV8Anchors`, `decodeYoloSeg`/`decodeYoloV8Seg`.

!!! note "Fonte da verdade"
    As assinaturas completas vivem no código-fonte em
    [`sdk-js-web/src/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-js-web/src).
    Esta página resume a superfície pública exportada em `index.ts`.
