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

!!! warning "Celular com pouca memória"
    O ORT copia o modelo para o heap WASM e aloca grafo e pesos **em cima** dessa
    cópia. Enquanto isso acontece, os bytes buscados pelo SDK também estão vivos
    no heap JS — um `.onnx` de 5 MB custa 5 MB + 5 MB + pesos no mesmo instante. O
    SDK lê os metadados **antes** de construir a sessão justamente para esse buffer
    morrer o quanto antes (0.5.1 — antes disso ele sobrevivia a toda a construção).

    Num aparelho que ainda assim não fecha a conta, o ORT desiste com
    `Can't create a session. failed to allocate a buffer of size N`. Duas saídas,
    na ordem: carregue um modelo por vez (nunca dois `create` concorrentes) e
    libere o que não está em uso com `session.release()`; se não bastar, passe
    `readMetadata: false` **com `labels` explícito** — aí o ORT busca o modelo
    sozinho e nada do SDK segura os bytes. O tamanho de entrada continua vindo do
    grafo; só os nomes das classes se perdem.

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

## Saindo da main thread (`env.wasm.proxy`)

O backend WASM roda na thread que chamou ele. Como essa é a main thread, tanto a
criação da sessão quanto cada `predict()` **travam a interface** enquanto rodam —
e não é pouco: medido num desktop de 32 núcleos, um `warmup()` do detector +
classificador congelou a página por **805 ms**; num celular de 4 núcleos / 2 GB,
uma análise leva de 50 a 103 s.

O ONNX Runtime resolve isso com uma flag: `env.wasm.proxy`. Com ela ligada, o ORT
cria um Web Worker próprio (`onnxruntime-web-proxy-worker`) e manda create, run e
release por `postMessage`. O mesmo warmup medido acima: pior frame de **18 ms**,
zero frames acima de 50 ms.

Ligue **uma vez, antes da primeira sessão**:

```typescript
import { env } from "onnxruntime-web";
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

env.wasm.proxy = true; // antes do primeiro Detector.create / Classifier.create

const det = await Detector.create("/models/yolov8n.onnx", { providers: ["wasm"] });
const result = (await det.predict("/images/img.jpg"))[0];
for (const d of result) console.log(d.className, d.confidence, d.bbox.asXyxy());
```

!!! warning "Antes da primeira sessão, não depois"
    O ORT lê `env.wasm` quando inicializa o runtime WASM, e isso acontece dentro
    do primeiro `InferenceSession.create`. Setar a flag depois disso é ignorado
    em silêncio — a inferência volta pra main thread sem nada avisar.

!!! note "O worker não deixa nada mais barato"
    Ele muda *onde* o custo é pago, não o quanto. O heap WASM e a reserva de
    memória compartilhada do build pthread apenas mudam de thread; um aparelho que
    não consegue criar a sessão na main thread também não consegue no worker.

??? info "Detalhe técnico: por que isso precisou de um fix no SDK (0.7.1)"
    O proxy posta os tensores de entrada com os `ArrayBuffer`s na *transfer list*,
    o que **destaca** o buffer do lado de quem enviou. Como `LetterboxPipeline` e
    `ResizePipeline` guardam um `Float32Array` e reentregam o mesmo a cada `run()`,
    a predict seguinte escrevia num buffer destacado — silenciosamente com
    `length === 0` — e o ORT rejeitava com
    `Tensor's size(1228800) does not match data length(0).` a cada duas chamadas.
    Desde a 0.7.1 o buffer é substituído quando foi destacado. Em versões
    anteriores, `env.wasm.proxy` não funciona com as tasks embutidas.

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
