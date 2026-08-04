# Guia Web (browser)

Detalhes específicos do pacote TypeScript
`@mauriciobenjamin700/ort-vision-sdk-web`. A API espelha a do Python; as
diferenças relevantes são listadas abaixo.

## Criação assíncrona

No navegador, carregar o modelo é assíncrono — por isso as tarefas são criadas
com `await Task.create(...)` em vez de um construtor, e `predict()` é sempre
`async`:

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

const det = await Detector.create("/models/yolov8n.onnx");
const result = (await det.predict("/images/street.jpg"))[0];
```

Assim como no Python, `predict()` retorna uma lista de comprimento 1
(`Promise<DetectionResults[]>`); use `[0]`. Cada tarefa também expõe um alias
`run()` (paridade com `nn.Module.__call__` do PyTorch).

## Entradas aceitas

`predict(image)` e `loadImage(image)` aceitam:

- `string` — uma URL buscada via `fetch()`.
- `Blob` / `File` — para uploads de `<input type="file">`.
- `HTMLImageElement` — uma tag `<img>` existente.
- `HTMLCanvasElement` / `OffscreenCanvas` — canvas já renderizado.
- `ImageBitmap` — de `createImageBitmap()`.
- `ImageData` — buffer de pixels cru (RGBA do `getImageData()` do canvas).
- `RGBImage` — o wrapper canônico HWC RGB `Uint8Array` do SDK.

## Resolução de entrada

O `inputSize` é opcional e serve de fallback: a resolução vem do shape que o
grafo declara.

```typescript
const clf = await Classifier.create("/models/classify.onnx", { labels: LABELS });
console.log(clf.inputSize); // [224, 224] — lido do .onnx, não configurado
```

!!! danger "Por que isso importa"
    Um export `-cls` do Ultralytics sai em 224×224 e um detector em 640×640.
    Alimentar o grafo errado faz o ORT abortar com `Got invalid dimensions for
    input: images ... Got: 640 Expected: 224` — e o número só existe dentro do
    arquivo, então nenhuma configuração podia acertar sozinha.

Passar um `inputSize` que contradiz um grafo estático emite um aviso no console
e é ignorado (o ORT rejeitaria de todo jeito). Em modelos de eixo dinâmico o seu
valor vale, com `[224, 224]`/`[640, 640]` como último recurso. Ver
[O modelo manda](modelo.md).

```typescript
console.log(clf.session.inputShape); // [1, 3, 224, 224] — null em eixo dinâmico
await clf.session.release();         // libera a sessão nativa
```

## Rótulos

**`labels` é opcional: sem ele, valem os nomes que o modelo declara.** O
Ultralytics grava `names` nos metadados do `.onnx`, e uma lista mantida à mão do
lado pode ser reordenada por acidente — nada falha, as predições só trocam de
classe.

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

const det = await Detector.create("/models/detect.onnx");
console.log(det.labels); // ["ocular-mucosa"] — do modelo, não de um preset
console.log(det.numClasses); // 1 — deduzido do shape de saída (B, 4 + nc, N)
```

!!! check "Isso também conserta um tropeço antigo"
    Antes, um detector de uma classe **falhava** sem `labels` explícito: o
    default era o preset COCO de 80 nomes, que discordava da contagem de classes
    do modelo.

!!! note "De onde vêm os metadados no navegador"
    O `onnxruntime-web` não expõe o mapa de metadados do modelo — diferente do
    `custom_metadata_map` do Python. O SDK lê os `metadata_props` dos próprios
    bytes do `.onnx` no carregamento, e por isso passa a buscar o modelo ele
    mesmo quando você informa uma URL (é o mesmo download, só quem faz muda).
    Para manter o caminho antigo, passe `readMetadata: false`. Um arquivo
    truncado ou inesperado resulta em mapa vazio, nunca em erro.

A precedência é a mesma do Python: o que você passa ganha, depois os `names` do
modelo, e por último o preset:

```typescript
import { Detector, Classifier, COCO_CLASSES } from "@mauriciobenjamin700/ort-vision-sdk-web";

// 1) Preset embutido
const det = await Detector.create("/models/yolov8n.onnx", { labels: "coco" });

// 2) Lista explícita
const clf = await Classifier.create("/m.onnx", { labels: ["cat", "dog", "fox"] });

// 3) Dict esparso — lacunas viram "class_<id>"
const clf2 = await Classifier.create("/m.onnx", { labels: { 0: "cat", 2: "fox" } });

// 4) null — auto-gera "class_0", "class_1", ... (informe numClasses)
const clf3 = await Classifier.create("/m.onnx", { labels: null, numClasses: 1000 });
```

## Execution providers

A ordem de providers padrão é `["webgpu", "wasm"]` — o ONNX Runtime tenta WebGPU
primeiro e cai silenciosamente para WebAssembly se o WebGPU não estiver
disponível. Você pode sobrescrever por tarefa:

```typescript
const clf = await Classifier.create(model, {
  labels,
  providers: ["wasm"], // força CPU
});
```

Para o WebGPU realmente engajar, você precisa de um build recente do ORT-Web, um
navegador Chromium com WebGPU habilitado e um contexto seguro (`https://` ou
`localhost`) — ou os cabeçalhos COOP/COEP corretos se também quiser threading
wasm baseado em `SharedArrayBuffer`.

## Resultados

Os formatos de resultado espelham os do Python:

- `result.boxes` — visão em massa (`xyxy`, `xywh`, `xyxyn`, `xywhn`, `cls`,
  `conf`, `data`).
- `result.probs` (classificação) — `top1`, `top5`, `top1conf`, `top5conf`,
  `data`.
- `result.masks` (segmentação) — `data`, `xyxy`.
- Iterar o envelope produz objetos por instância com `classId`/`className`/
  `confidence`/`bbox` e os aliases `cls`/`name`/`conf`/`box`. A `BoundingBox`
  expõe `asXyxy()` e `asXywh()`.

## Veja também

- [Referência da API Web](../referencia/web.md)
- [Guia Python](python.md) — o equivalente no backend.
