# Referência — API Web

Superfície pública do pacote `@mauriciobenjamin700/ort-vision-sdk-web` (tudo
exportado de `@mauriciobenjamin700/ort-vision-sdk-web`).

## Tarefas

| Classe | Criação | Descrição |
| --- | --- | --- |
| `Classifier` | `await Classifier.create(model, options)` | Classificação de imagem. |
| `Detector` | `await Detector.create(model, options)` | Detecção de objetos (YOLO). |
| `Segmenter` | `await Segmenter.create(model, options)` | Segmentação de instância (YOLO-seg). |
| `DetectClassify` | `await DetectClassify.create(model, options)` | Pipeline fundido detector → classificador, num único `.onnx`. |
| `VisionTask` | — | Classe base comum. |

`predict()` é sempre `async` e retorna `Promise<...Results[]>` de comprimento 1
por imagem. Cada tarefa expõe um alias `run()`.

### Tipos de opções

| Tipo | Para |
| --- | --- |
| `ClassifierOptions` / `ClassifierPredictOptions` | construção / `predict` do `Classifier` (`labels`, `numClasses`, `inputSize`, `normalization`, `mean`/`std`, `applySoftmax`, `providers`; `topK` no predict) |
| `DetectorOptions` / `DetectorPredictOptions` | `Detector` (`head`, `labels`, `inputSize`, `confThreshold`, `iouThreshold`; overrides + `classes` no predict) |
| `SegmenterOptions` / `SegmenterPredictOptions` | `Segmenter` (+ `maskThreshold`) |
| `DetectorHead` (`"yolo"`) / `SegmenterHead` (`"yolo-seg"`) | famílias de decoder |
| `DetectClassifyOptions` / `DetectClassifyPredictOptions` | `DetectClassify` (`labels`, `classifierLabels`; `confThreshold`, `classes`, `topK` no predict) |

Os três tipos de detecção aceitam ainda `raiseOnEmpty` (construção e predict) —
ver [Resultado vazio](#resultado-vazio).

## Resultados

| Envelope | Visão em massa | Iterar produz |
| --- | --- | --- |
| `ClassificationResults` | `probs` | n/a (resultado único) |
| `DetectionResults` | `boxes` | `DetectionResult` |
| `DetectClassifyResults` | `boxes` | `DetectionResult` com `classification` preenchido (+ `classifierNames` no envelope) |
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
| `detectProviders(...)` | Estreita a lista pedida pelo que o navegador consegue oferecer (WebGPU precisa de adapter). |
| `Normalization` / `resolveNormalization(...)` / `isUltralyticsClassifier(...)` | Qual preprocessamento o classificador espera, lido dos metadados do modelo. |
| `OrtSession` / `OrtSessionOptions` / `ModelSource` | Sessão de baixo nível. |
| `OrtSession.inputShape` / `.inputShapes` | Shapes declarados pelo grafo, eixos dinâmicos como `null`. |
| `OrtSession.providers` | Providers que este navegador pode oferecer — best-effort, o ORT-Web não reporta o efetivo. |
| `OrtSession.requestedProviders` | Providers que foram pedidos, depois dos defaults. |
| `OrtSession.release()` | Libera a sessão nativa (necessário ao descartar uma sessão com a página viva). |
| `task.inputSize` | Resolução em que a tarefa realmente pré-processa. |
| `task.warmup(runs?)` | Roda o modelo com tensor zerado para pagar a compilação de shaders adiantado. |
| `spatialInputSize` / `resolveInputSize` / `declaredShapesFrom` | Helpers puros da precedência grafo → chamador → fallback. |
| `DeclaredShape` / `DeclaredDim` | Shape declarado e uma dimensão (`number`, ou `null` quando simbólica). |

## Erros

Hierarquia de exceções exportada: `OrtVisionError` (base), `ImageLoadError`,
`InferenceError`, `LabelMapError`, `ModelLoadError`,
`ProviderNotAvailableError`, `FusionError`, `NoDetectionsError`.

## Resultado vazio

`Detector`, `Segmenter` e `DetectClassify` aceitam `raiseOnEmpty` nas opções de
construção e nas de `predict()`. Default `false`: não achar nada devolve um
envelope vazio. Com `true`, lança `NoDetectionsError` — ver
[Quando não detectar nada é um erro](../guia/deteccao.md#quando-nao-detectar-nada-e-um-erro).

| Símbolo | Descrição |
| --- | --- |
| `raiseOnEmpty` | Opção de construção e de `predict()`; o valor por chamada vence. |
| `NoDetectionsError` | Lançado quando nada sobra e o flag está ativo. |
| `requireDetections(count, options)` | O helper compartilhado pelas três tarefas, exportado para quem constrói a própria tarefa. |

## Pipelines fundidos

| Símbolo | Descrição |
| --- | --- |
| `readFusionSpec(metadata)` | Lê o que um pipeline fundido declara sobre si mesmo; `null` quando o modelo não é um pipeline. |
| `FusionSpec` / `CropSource` | O contrato decodificado e de onde vêm os recortes. |
| `INPUT_IMAGE` / `INPUT_SOURCE` / `INPUT_SCALE` / `INPUT_PAD` | Nomes das entradas do grafo fundido. |
| `OUTPUT_BOXES` / `OUTPUT_SCORES` / `OUTPUT_CLASSES` / `OUTPUT_NUM_DETECTIONS` / `OUTPUT_PROBS` | Nomes das saídas. |
| `METADATA_PREFIX` / `FUSION_KIND_DETECT_CLASSIFY` | Namespace `ovs.` e a família de pipeline. |
| `parseNames(raw)` | Interpreta um mapa de classes `repr`-encoded. |

Fundir modelos é um passo de build **do lado Python** (extra `[compose]`); o
navegador só carrega o `.onnx` resultante. Ver
[Pipelines fundidos](../guia/pipeline.md).

## Utilitários de pré/pós-processamento

O pacote também exporta helpers de baixo nível para quem constrói o próprio
pipeline: `letterbox`, `resize`, `normalize`, `toCHW`, `toTensor`,
`toFloat32`/`toFloat32Tensor`, `fromCv2`/`toCv2`, `softmax`, `topK`, `nms`,
`batchedNms`, `decodeYolo`, `decodeYoloAnchors` e `decodeYoloSeg`.

### O caminho rápido que as tarefas tomam

Os primitivos acima alocam e varrem o buffer inteiro a cada chamada — a forma
certa para uma biblioteca, a errada para um laço de vídeo. As tarefas internas
usam duas pipelines que fundem esse trabalho num `drawImage` mais um laço, com
o buffer de saída reusado entre frames:

| Símbolo | O que faz |
| --- | --- |
| `LetterboxPipeline(w, h, fill?)` | Redimensiona **preservando proporção** e preenche o resto, devolvendo `{ data, scale, padLeft, padTop, reused }`. É o que `Detector`, `Segmenter` e `DetectClassify` usam. |
| `ResizePipeline(w, h, mean?, std?)` | Estica até o alvo (sem padding) e já normaliza, devolvendo `{ data, reused }`. É o que o `Classifier` usa — ele não mapeia nada de volta para a imagem original, então não há escala nem padding a inverter. |
| `letterboxToTensorData(...)` / `resizeToTensorData(...)` | As formas de uma chamada só, para quem não quer manter a pipeline viva. |
| `writePlanarFloat32(rgba, w, h, mean, std, out, stride?)` | O laço em si: RGBA (ou RGB empacotado, com `stride: 3`) → float32 planar normalizado. |
| `zeroTensorData(w, h)` | O tensor zerado que o `warmup()` alimenta. |

!!! warning "`release()` não é opcional"
    O buffer de saída é reusado, então `run()` marca-o como em uso e a chamada
    seguinte aloca outro em vez de corromper o primeiro. Chame `release()`
    depois que a inferência resolveu — a partir daí os valores já estão dentro
    do heap do WASM.

!!! info "A saída é bit-idêntica à dos primitivos"
    Fundir mudou quantas passadas e quantas alocações acontecem, não a
    aritmética: `(valor / 255 - mean) / std` é avaliado nessa ordem justamente
    porque colapsar numa multiplicação-e-soma daria outro arredondamento. Os
    testes comparam as duas saídas valor a valor.

!!! note "Fonte da verdade"
    As assinaturas completas vivem no código-fonte em
    [`sdk-js-web/src/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-js-web/src).
    Esta página resume a superfície pública exportada em `index.ts`.
