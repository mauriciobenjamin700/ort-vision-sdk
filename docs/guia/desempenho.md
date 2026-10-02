# Otimizando a criação e a inferência

Na página anterior você aprendeu a [medir](velocidade.md) para onde vai o
tempo. Esta página é o passo seguinte: **o que fazer com esse número.** 🚀

Há dois custos diferentes, e cada um tem as suas alavancas:

- **Criar a tarefa** (`Detector.create(...)` / `Detector(...)`): baixar o
  modelo e montar a sessão do ONNX Runtime. Acontece uma vez, mas é o que o
  usuário espera antes de ver qualquer coisa.
- **Cada `predict()`**: decodificar a imagem, pré-processar, rodar o modelo,
  decodificar a saída. Acontece a cada frame.

!!! info "Todo número aqui foi medido"
    Salvo indicação em contrário, os números vêm de um Chromium headless
    (Playwright) numa máquina de 12 núcleos lógicos, backend WASM, com um
    YOLO11n-seg 640×640 (11,7 MB). Cada valor é a **mediana de 5 a 12
    execuções** depois de uma execução de aquecimento. Num celular os números
    absolutos são outros; as proporções costumam se manter.

## Criação: guarde o modelo no navegador

Num carregamento de página, o maior custo da criação quase sempre é **baixar o
modelo**. São megabytes, e o servidor nem sempre manda cabeçalhos de cache que
o navegador respeite para um arquivo desse tamanho.

A opção `cache` guarda o modelo na Cache Storage do navegador. Da segunda visita
em diante, os bytes vêm do disco, sem rede:

```typescript hl_lines="4"
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

const det = await Detector.create("/models/yolov8n.v1.onnx", {
  cache: true,
});
const result = (await det.predict("/images/street.jpg"))[0];
for (const d of result) console.log(d.className, d.confidence, d.bbox.asXyxy());
```

`cache: true` usa o bucket `DEFAULT_MODEL_CACHE` (`"ort-vision-sdk-models"`).
Uma string escolhe outro nome: `cache: "meu-app-modelos"`.

!!! warning "A URL é a chave, e nada expira sozinho"
    Se você publicar um modelo novo **na mesma URL**, quem já visitou continua
    com o antigo. Versione a URL (`yolov8n.v2.onnx`, `?v=2`), troque o nome do
    bucket, ou apague o bucket com `caches.delete(nome)`.

!!! note "Onde não há Cache Storage, nada quebra"
    A Cache Storage só existe em contexto seguro (`https://` ou `localhost`) e
    pode estar bloqueada (aba anônima, dados do site desativados). Nesses casos
    o SDK simplesmente baixa o modelo da rede, como se `cache` não tivesse sido
    pedido.

## Criação: baixe o runtime junto com o modelo

Além do modelo, a primeira sessão de uma página baixa e compila o **runtime do
ONNX Runtime**, um `.wasm` de vários megabytes. Antes, isso só começava depois
do download do modelo terminar, e os dois maiores downloads da página
aconteciam um depois do outro.

Agora o SDK sobe o runtime **enquanto** o modelo baixa. Você não muda nada.
Medido com a banda limitada a 5 MB/s (modelo de 11,7 MB, runtime de 12,8 MB),
na primeira criação da página:

| | Primeira criação |
| --- | --- |
| Antes (um depois do outro) | 5020–5145 ms |
| Agora (em paralelo) | **2648–2656 ms** |

### Não usa WebGPU? Baixe metade do runtime

O import padrão `onnxruntime-web` traz o runtime **com** WebGPU: 25,9 MB (6,0 MB
com gzip). O subpath `onnxruntime-web/wasm` traz só o WASM: 12,8 MB (3,3 MB com
gzip). Se o seu app roda só em WASM, aponte o import para ele. No Vite, em
`vite.config.ts`:

```typescript
import { defineConfig } from "vite";

export default defineConfig({
  resolve: {
    alias: [{ find: /^onnxruntime-web$/, replacement: "onnxruntime-web/wasm" }],
  },
});
```

O SDK também passa a usar esse build, porque ele importa `onnxruntime-web`. Peça
`providers: ["wasm"]` nas tarefas: o build só WASM não tem WebGPU para oferecer.

## Criação: otimize o grafo uma vez, no build

Toda sessão do ONNX Runtime roda o **otimizador de grafo** antes de inferir:
dobra constantes, funde operadores, remove nós redundantes. É a maior parte do
custo de montar a sessão, e ele dá o mesmo resultado toda vez.

Então dá para pagar isso **uma vez, no seu build**, com o SDK Python:

```python
from ort_vision_sdk import optimize_model

optimize_model("yolov8n.onnx", "yolov8n.opt.onnx")
```

Publique o `yolov8n.opt.onnx` no lugar do original. Pronto! O SDK web lê, na
metadata do arquivo, a marca que o `optimize_model` deixou e cria a sessão
**sem** rodar o otimizador de novo. Você não muda nenhuma linha no browser:

```typescript
import { Segmenter } from "@mauriciobenjamin700/ort-vision-sdk-web";

const seg = await Segmenter.create("/models/yolov8n-seg.opt.onnx");
```

| Arquivo | Criar a sessão (só ORT) | `Segmenter.create` completo | Inferência |
| --- | --- | --- | --- |
| Original | 29,4 ms | 52,6 ms | 65,9 ms |
| `optimize_model(...)` | **12,4 ms** | **33,6 ms** | 65,2 ms |

A inferência não muda: o grafo é o mesmo que o ORT montaria sozinho, só que
montado antes.

!!! tip "Por que `"extended"` e não `"all"`"
    O padrão é `level="extended"`. Além disso, o ORT aplica transformações de
    layout escolhidas para as instruções da **CPU que está otimizando**
    (AVX2, AVX-512…), e o WASM do navegador não tem essas instruções. Por isso
    `"all"` nem é oferecido. `"basic"` existe para quem quer só dobra de
    constantes.

??? info "Detalhes técnicos"
    - A metadata do export (`names`, `imgsz`…) é preservada; a marca é a chave
      `GRAPH_OPTIMIZATION_KEY` (`"ort_vision_sdk.graph_optimization"`).
    - Um `graphOptimizationLevel` explícito em `sessionOptions` sempre vence a
      marca.
    - Com `readMetadata: false` a marca não é lida, e o ORT otimiza normalmente.
    - As fusões do nível `extended` usam operadores `com.microsoft` que toda
      build de CPU e de WASM implementa. No **WebGPU**, parte deles não tem
      kernel: no YOLO11n-seg, o ORT colocou 2 nós na CPU para o arquivo
      otimizado e nenhum para o original, e isso custa cópias GPU↔CPU a cada
      inferência. O SDK avisa no console quando um modelo pré-otimizado vai
      para o WebGPU. **Para WebGPU, publique o export original.**
    - O SDK Python não aplica a marca sozinho: ele só lê a metadata depois que
      a sessão já existe. O arquivo otimizado roda normalmente; para pular a
      otimização repetida, passe
      `SessionOptions` com `graph_optimization_level = ORT_DISABLE_ALL`.

## Criação: opções por execution provider

Cada execution provider tem opções próprias, e algumas mudam muito o tempo de
criação e de inferência. Passe um objeto (web) ou um par `(nome, opções)`
(Python) no lugar do nome:

=== "Web"

    ```typescript hl_lines="4"
    import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const det = await Detector.create("/models/yolov8n.onnx", {
      providers: [{ name: "webgpu", preferredLayout: "NHWC" }, "wasm"],
    });
    console.log(det.session.providers); // ["webgpu", "wasm"] — sempre nomes
    ```

=== "Python"

    ```python hl_lines="6-7"
    from ort_vision_sdk import Detector

    det = Detector(
        "yolov8n.onnx",
        providers=[
            ("tensorrt", {"trt_engine_cache_enable": True, "trt_engine_cache_path": "./trt"}),
            ("cuda", {"cudnn_conv_algo_search": "HEURISTIC"}),
            "cpu",
        ],
    )
    print(det.session.requested_providers)
    # ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    ```

No Python, o nome do par aceita os mesmos apelidos de sempre (`"cuda"`,
`"tensorrt"`…). As opções vão intactas para o ONNX Runtime.

!!! tip "O cache de engine do TensorRT é o que mais rende"
    Sem cache, o TensorRT recompila o engine **a cada processo**, e para um
    modelo grande isso leva minutos. Com `trt_engine_cache_enable`, só o
    primeiro processo paga.

## Inferência: quantize para INT8

Quantizar troca os pesos e as ativações de float32 por inteiros de 8 bits. O
modelo fica ~3× menor, e a CPU passa a rodar kernels inteiros, bem mais
baratos. Também é um passo de build no SDK Python, com o extra `[quantize]`:

```python
from pathlib import Path

from ort_vision_sdk import quantize_model

calibration = sorted(Path("calibration/").glob("*.jpg"))
quantize_model("yolov8n-seg.onnx", "yolov8n-seg.int8.onnx", calibration)
```

As imagens de calibração definem a faixa int8 de cada ativação, e passam pelo
**mesmo pré-processamento** do `predict()`: letterbox para detecção e
segmentação, resize e normalização para classificação. A tarefa vem do campo
`task` da metadata do Ultralytics, ou de `task="detect" | "segment" |
"classify"`.

Medido no YOLO11n-seg (QDQ por canal, 24 imagens de calibração):

| | FP32 | INT8 |
| --- | --- | --- |
| Arquivo | 11,7 MB | **3,5 MB** |
| Inferência, CPU nativa (ORT Python) | 87 ms | **30,5 ms** |
| `predict()` completo, SDK Python | 37,5 ms | **16,8 ms** |
| Inferência, WASM (4 threads) | 68,4 ms | **52,7 ms** |

!!! warning "Valide a precisão nos seus dados"
    A quantização custa alguma precisão, e quanto custa depende do modelo e de
    quão representativas são as imagens de calibração. Neste modelo a detecção
    principal ficou igual (confiança 0,663 → 0,659), mas meça no seu conjunto de
    validação antes de publicar.

!!! note "No browser, INT8 roda em WASM"
    O backend WebGPU do ONNX Runtime Web não roda os operadores quantizados
    desse formato: o `DequantizeLinear` recusa o bias quantizado. O
    `quantize_model` marca o arquivo, e o SDK web tira o `webgpu` dos providers
    de um modelo marcado, com um aviso no console. Em WASM, que é onde o INT8
    compensa, ele roda normalmente.

??? info "Por que não encadear com `optimize_model`"
    Para modelo INT8, o ORT faz fusões **no load** que o nível `extended`
    offline não reproduz. Medido em WASM: o INT8 pré-otimizado foi criado 79 ms
    mais rápido, mas cada inferência ficou 19% mais lenta (60,8 contra
    51,1 ms). O `optimize_model` emite um `UserWarning` quando recebe um modelo
    quantizado.

## Inferência: threads do WASM

O backend WASM usa várias threads, mas **só** quando a página está isolada
(`crossOriginIsolated === true`). Sem isolamento, roda numa thread só. Medido no
YOLO11n-seg:

| `env.wasm.numThreads` | Inferência |
| --- | --- |
| 1 (página sem isolamento) | 221 ms |
| 4 (padrão do ORT) | 65 ms |
| **6** | **49 ms** |
| 10 | 58 ms |

Duas lições aqui:

1. **Isolar a página vale 3,4×.** É o maior ganho desta página inteira, e não
   muda uma linha do seu código: são dois cabeçalhos HTTP.
2. O padrão do ORT limita em 4 threads. Nesta máquina, **metade dos núcleos**
   foi o melhor; passar disso piorou.

Para isolar, sirva a página com:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

E, se quiser ajustar as threads, faça isso **antes da primeira sessão**:

```typescript
import { env } from "onnxruntime-web";
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

env.wasm.numThreads = Math.max(1, Math.floor(navigator.hardwareConcurrency / 2));

const det = await Detector.create("/models/yolov8n.onnx", { providers: ["wasm"] });
```

!!! warning "Por que o SDK não faz isso sozinho"
    `env.wasm` é configuração **global** do `onnxruntime-web`, que pertence ao
    seu app (é uma peer dependency), e o melhor número depende do aparelho e
    do que mais roda na página. O SDK não mexe nela por você.

!!! note "COEP tem um preço"
    Com `require-corp`, todo recurso de outra origem (imagem de CDN, fonte,
    script) precisa vir com `Cross-Origin-Resource-Policy` ou CORS. Confira
    antes de ligar em produção.

## Inferência: dois `predict` em voo

Com `env.wasm.proxy = true`, a inferência roda num worker, e a main thread
fica livre enquanto ela acontece. Dá para aproveitar esse tempo: enquanto o
modelo infere o frame N, o SDK já decodifica e pré-processa o frame N+1.

Basta manter **dois** `predict()` em andamento. O SDK enfileira as execuções da
mesma sessão (o ONNX Runtime Web recusa duas ao mesmo tempo), então só a etapa
do modelo espera a vez:

```typescript
import { env } from "onnxruntime-web";
import { Segmenter } from "@mauriciobenjamin700/ort-vision-sdk-web";

env.wasm.proxy = true;

const seg = await Segmenter.create("/models/yolov8n-seg.onnx", { providers: ["wasm"] });
await seg.warmup();

const frames: HTMLCanvasElement[] = [...document.querySelectorAll("canvas")];
let next = 0;
async function worker(): Promise<void> {
  while (next < frames.length) {
    const frame = frames[next++]!;
    const result = (await seg.predict(frame))[0];
    console.log(result.length, "instâncias");
  }
}
await Promise.all([worker(), worker()]);
```

Medido num frame 1080p, com proxy:

| `predict()` em voo | ms por frame |
| --- | --- |
| 1 | 82–86 |
| **2** | **70–73** |
| 3 | 72 |

Três não ganham mais nada: o gargalo passa a ser o próprio modelo. Sem
`env.wasm.proxy`, a inferência ocupa a main thread e dois em voo não ganham
nada (84,6 contra 86 ms), mas também não quebram.

## Inferência: entregue o vídeo direto

Num loop de câmera, passe o `HTMLVideoElement` (ou um `VideoFrame`) direto para
o `predict()`, sem desenhar num canvas antes:

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

const video = document.querySelector("video")!;
video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
await new Promise((resolve) => video.addEventListener("loadeddata", resolve, { once: true }));
await video.play();

const det = await Detector.create("/models/yolov8n.onnx", { cache: true });
await det.warmup();

async function loop(): Promise<void> {
  const result = (await det.predict(video))[0];
  console.log(result.length, "objetos", result.speed);
  requestAnimationFrame(loop);
}
loop();
```

O frame decodificado é o que está na tela no momento da chamada. Um vídeo sem
frame ainda (`readyState < 2`) lança `ImageLoadError` explicando que é preciso
esperar o evento `loadeddata`.

Com câmera ou tela (`srcObject` é um `MediaStream`), o SDK nem lê o frame de
volta da GPU: desenha numa cópia e pré-processa a partir dela. Os pixels em RGB
só são montados se alguém ler `origImg`, `croppedImage` ou `segmentedImage`.
Medido num frame 1080p, o `load` caiu de 8,5–10,7 ms para **2,0–2,5 ms**. Vale
também para `VideoFrame` sem alpha (`I420`, `NV12`, `RGBX`…) e para JPEG.

??? info "Por que só essas fontes"
    Pré-processar direto da cópia só é idêntico ao caminho normal quando
    nenhum pixel é translúcido, e normalmente só lendo os pixels dá para saber.
    Frame de `MediaStream`, `VideoFrame` sem alpha e JPEG **não têm como**
    carregar alpha, então dispensam a leitura. Um `<video>` tocando arquivo
    pode ter alpha (VP9), e por isso continua sendo lido na hora.

## O que o SDK já faz por você

Algumas otimizações não pedem nada de você. Vale saber que existem, porque elas
explicam números do `speed`:

- **Pré-processamento sem cópia extra.** Quando a entrada é desenhável
  (imagem, canvas, `ImageBitmap`, vídeo, `Blob`, URL), o pipeline redimensiona
  direto do canvas em que a imagem foi decodificada. Medido num frame 1080p, o
  `preprocess` caiu de ~10 ms para ~5 ms. Imagem com **transparência** usa o
  caminho antigo, porque compor o alpha mudaria as cores.
- **`croppedImage` e `segmentedImage` sob demanda.** Os recortes por detecção
  só são montados quando você lê o campo. Se você só usa caixa e classe, eles
  nunca custam nada.
- **Pós-processamento mais rápido**, com saída idêntica bit a bit à anterior:
  decode 25–54% mais rápido, NMS 28–40%, montagem de máscara 71% no web; no
  Python, decode 43–80%, NMS 78–87% (até 512 caixas por classe) e máscara 81%.
- **`warmup()` nos dois SDKs.** No Python ele paga antes da primeira requisição
  a alocação do CUDA, a busca de algoritmo do cuDNN e o engine do TensorRT.

!!! warning "Reaproveita o mesmo `RGBImage` entre frames? Os recortes ficam imediatos"
    Se você passa um `RGBImage` seu e reescreve o buffer dele a cada frame, um
    recorte montado depois leria o frame seguinte. Por isso, com `RGBImage` do
    chamador, o SDK monta os recortes **na hora**. O modo sob demanda vale só
    quando o próprio SDK alocou os pixels.

## Recapitulando

- **Criação:** `cache: true` tira a rede do caminho a partir da segunda visita;
  o runtime agora baixa junto com o modelo (−48% na primeira criação);
  `onnxruntime-web/wasm` corta o runtime pela metade; `optimize_model` no build
  corta ~60% da montagem da sessão (para WASM); opções por provider via objeto
  ou par `(nome, opções)`. ✅
- **Inferência:** `quantize_model` faz o modelo ~3× menor e 2,2× mais rápido no
  Python; isole a página (COOP/COEP) para ganhar até 3,4× no WASM; com
  `env.wasm.proxy`, dois `predict()` em voo rendem ~15% a mais.
- **Vídeo:** passe o `HTMLVideoElement` direto para o `predict()`; com câmera,
  o `load` cai ~75%.
- O SDK já evita cópias no pré-processamento e monta os recortes só quando
  você pede.

## Veja também

- [Custo da inferência](velocidade.md): medir antes de otimizar.
- [Guia Web](web.md): `env.wasm.proxy` para tirar a inferência da main thread.
