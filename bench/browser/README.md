# Bancada de browser

Mede um modelo no navegador de verdade: WASM e WebGPU, quatro variantes do mesmo
export (original, pré-otimizado, INT8, FP16), o `predict()` do SDK por etapa
num frame 1080p, e o ganho de manter dois `predict()` em voo. Complementa os
microbenchmarks de `scripts/bench.py` e `scripts/bench_web.mjs`, que medem só
as funções puras e não tocam em canvas, ONNX Runtime ou GPU.

A página roda sozinha no navegador — sem Playwright — e devolve um JSON. Os
modelos são seus e ficam em `bench/browser/models/`, que o git ignora.

## 1. Preparar o ambiente

```bash
npm --prefix sdk-js-web ci
uv venv sdk-python/.venv
uv pip install -p sdk-python/.venv -e "sdk-python[dev]"
```

## 2. Gerar as variantes do modelo

```bash
make bench-browser-models MODEL=caminho/yolo11n-seg.onnx IMAGE=caminho/frame.jpg \
    CALIBRATION=caminho/imagens_de_calibracao/
```

Escreve em `bench/browser/models/`:

| Arquivo | Origem |
| --- | --- |
| `model.onnx` | cópia do export |
| `model.opt.onnx` | `optimize_model` (otimização de grafo offline) |
| `model.int8.onnx` | `quantize_model` (INT8 QDQ por canal) |
| `model.fp16.onnx` | FP16, com entrada e saída em float32 |
| `image.jpg` | a imagem do `predict()` |

Sem `CALIBRATION`, o INT8 é calibrado só com a imagem do teste: serve para medir
tempo, não para julgar precisão.

## 3. Servir e abrir

```bash
make bench-browser
```

Abra `http://localhost:8765/bench/browser/run.html` e deixe a aba visível até
aparecer **PRONTO ✅**. O JSON vai sozinho para `bench/browser/results/`, e o
botão **Copiar JSON** continua valendo.

Se aparecer **PRONTO ⚠️**, a aba ficou oculta em algum momento e os tempos não
valem — veja *A aba precisa estar visível* abaixo.

O servidor manda `Cross-Origin-Opener-Policy` e `Cross-Origin-Embedder-Policy`
(sem eles o WASM roda numa thread só) e desliga o cache HTTP.

### Num celular Android

Com o celular pareado por depuração (USB ou Wi-Fi):

```bash
adb reverse tcp:8765 tcp:8765
```

E abra `http://localhost:8765/bench/browser/run.html` no Chrome do celular. Tem
de ser `localhost`: é o que conta como contexto seguro, que WebGPU e Cache
Storage exigem. Deixe o aparelho **na tomada** e a aba **visível** — fora da
tomada o sistema segura CPU e GPU, e aba em segundo plano congela.

### No Windows com WSL

Com a rede espelhada do WSL, o `localhost` do Windows alcança o servidor que
roda no WSL. O Chrome do Windows usa a GPU real, que o WSL não enxerga.

## Parâmetros da página

| Parâmetro | Efeito |
| --- | --- |
| `?task=segment` / `detect` / `classify` | Classe do SDK usada (padrão `segment`) |
| `?size=640` | Lado do tensor do teste de ORT puro (padrão 640) |
| `?threads=N` | `env.wasm.numThreads` — recarregue a página para cada valor |
| `?proxy=1` | `env.wasm.proxy`: inferência num worker; é onde dois `predict()` em voo ganham |
| `?quick=1` | Menos repetições |
| `?only=breakdown` | Roda só as seções listadas, separadas por vírgula: `raw`, `sdk`, `breakdown`, `pipelining` (`compare` roda sempre que pedido) |
| `?set=nome` | Lê os modelos de `bench/browser/models/<nome>/` — gerado com `prepare_browser_bench.py --set <nome>` |
| `?compare=a,b` | Compara conjuntos na mesma página, em rounds alternados; o primeiro é a base do `speedup_*` |
| `?rounds=4` | Rounds da comparação (padrão 4; 2 com `?quick=1`) |

## Compare modelos na mesma página

No celular, o clock muda entre uma página e outra. Num Mali Bifrost, o WebGPU
variou 2,4× entre rodadas abertas com minutos de diferença, e o número de
threads do WASM nem chega nele. Comparar `det640` numa página e `det320` em
outra mede o estado do aparelho, não os modelos.

`?compare=` carrega os conjuntos juntos e reveza a cada round (A B, B A, A B…),
então a deriva atinge os dois igualmente. `inferencePerRound` vai no JSON: se
os rounds de um mesmo modelo discordam muito, a deriva foi grande, e a razão
`speedup_*` continua valendo; os valores absolutos, não.

```text
run.html?task=detect&compare=famacha-det640,famacha-det320&only=compare
```
| `?ramp=400` | Milissegundos de inferência descartada antes de cronometrar, para o clock subir |
| `?timeout=180000` | Teto por medição; o que estourar vira `error` no JSON e a bateria segue |
| `?label=nome` | Nome do arquivo em `bench/browser/results/` |

## Quebra do quadro (`breakdown`)

O `speed` do SDK junta desenho, readback e normalização em `preprocess`, e
upload, modelo e download em `inference`. A seção `breakdown` refaz à mão o
caminho rápido do SDK num frame 1080p e cronometra cada fronteira de cópia à
parte. É o número que decide se vale pré-processar dentro do grafo, reduzir a
saída dentro do grafo ou manter a entrada e a saída na GPU.

| Campo | Mede |
| --- | --- |
| `inputBytes` / `outputBytes` | Tamanho do tensor de entrada e soma das saídas |
| `drawReadbackMs` | Letterbox no canvas `willReadFrequently` (o que o SDK usa) + `getImageData` |
| `drawReadbackGpuCanvasMs` | O mesmo num canvas acelerado, para comparar |
| `canvasPixelDiff` | Diferença de pixel entre os dois canvas acima (`maxAbs`, `meanAbs` em níveis de 0–255, `changedFraction` dos canais). Acelerado e software podem reamostrar com filtros diferentes: é o que o modelo passa a ver se o SDK trocar de um para o outro |
| `normalizeMs` | Laço RGBA → CHW float32 |
| `runCpuIoMs` | `run` com entrada e saída na CPU: upload + modelo + download |
| `runGpuOutMs` | Só WebGPU. `run` com saída `gpu-buffer`, até a promessa resolver |
| `gpuDrainMs` | Só WebGPU. Espera a GPU terminar (`onSubmittedWorkDone`) |
| `downloadMs` | Só WebGPU. `getData()` das saídas: mapear e copiar de volta |
| `runGpuIoMs` | Só WebGPU. Entrada já na GPU (`Tensor.fromGpuBuffer`) até a GPU terminar |
| `downloadGpuIoMs` | Só WebGPU. Download das saídas dessa execução |
| `runGraphMs` | Só WebGPU. Igual a `runGpuIoMs`, com `enableGraphCapture`: a GPU reexecuta comandos gravados, sem a CPU despachar kernel por kernel |
| `downloadGraphMs` | Só WebGPU. Download das saídas da execução gravada |
| `graphMaxAbsDiff` | Maior diferença entre a saída gravada e a normal, para a mesma entrada. Tem de ser ~0 |
| `graphCaptureError` | Por que o graph capture não rodou neste modelo, quando não rodou |

Leituras:

- `runCpuIoMs − runGpuIoMs − downloadGpuIoMs` ≈ custo do upload da entrada.
- `downloadMs` é o teto do que se ganha reduzindo a saída dentro do grafo.
- `drawReadbackMs + normalizeMs` é o teto do que se ganha pré-processando
  na GPU.
- `runGpuIoMs − runGraphMs` é o que o graph capture tira do despacho pela CPU.
  Num celular (Mali Bifrost, det 320), o despacho sozinho (`runGpuOutMs`) foi
  128 de ~224 ms de inferência.

> **`runGpuOutMs` sozinho não diz nada.** Com saída `gpu-buffer`, a promessa do
> `run` resolve antes de a GPU terminar. Num adapter de software, ela resolveu
> em 25 ms e a GPU ainda levou 12 s. Some `runGpuOutMs + gpuDrainMs`.

O Chrome adia o `drawImage` e só o executa no `getImageData` seguinte. Por isso
desenho e readback saem num número só: medido à parte, o desenho dava 0 ms.

Para rodar só esta seção, que é rápida, use
`run.html?only=breakdown`.

## A aba precisa estar visível

O Chrome calcula oclusão nativa: janela **coberta por outra** conta como oculta,
e aba oculta faz ele suspender o trabalho de WebGPU. O efeito barulhento é a
bateria travar na criação de uma sessão WebGPU. O efeito perigoso é mais sutil —
às vezes ela só fica lenta, e o JSON sai plausível e errado: numa RTX 4070 Ti
SUPER o mesmo `webgpu_fp32` deu 12,7 ms com a aba visível e 37,8 ms com ela
oculta, sem nada no resultado denunciando a diferença.

Por isso `env.visibility` e `env.everHidden` vão no JSON, e a página termina em
**PRONTO ⚠️** quando a aba piscou para oculta. Descarte essas rodadas.

Para medir numa máquina que você está usando, sem disputar o primeiro plano,
suba uma instância dedicada do Chrome com a oclusão desligada:

```bash
chrome --user-data-dir=/tmp/bench-profile        --disable-features=CalculateNativeWinOcclusion        --disable-backgrounding-occluded-windows        --disable-renderer-backgrounding        "http://localhost:8765/bench/browser/run.html"
```

O perfil separado mantém o seu Chrome fora da medição, sem extensão nenhuma
rodando junto. No Android não há como passar flag: lá o `everHidden` é o que
avisa.

## Simular rede lenta

```bash
node bench/browser/server.mjs --bandwidth 5
```

## Onde o resultado cai

O servidor grava o POST da página em `bench/browser/results/<label>.json` (o git
ignora o diretório). `--results ""` desliga a gravação.

Atrasa cada `.onnx` e `.wasm` pelo tamanho, a 5 MB/s, para medir a primeira
criação de uma página como um visitante novo vê.
