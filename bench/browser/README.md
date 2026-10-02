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
aparecer **PRONTO ✅**. **Copiar JSON** leva o resultado.

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

## Simular rede lenta

```bash
node bench/browser/server.mjs --bandwidth 5
```

Atrasa cada `.onnx` e `.wasm` pelo tamanho, a 5 MB/s, para medir a primeira
criação de uma página como um visitante novo vê.
