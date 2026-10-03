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
| `?ramp=400` | Milissegundos de inferência descartada antes de cronometrar, para o clock subir |
| `?timeout=180000` | Teto por medição; o que estourar vira `error` no JSON e a bateria segue |
| `?label=nome` | Nome do arquivo em `bench/browser/results/` |

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
