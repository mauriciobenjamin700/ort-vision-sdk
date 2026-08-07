# Instalação

O `ort-vision-sdk` é distribuído como dois pacotes independentes. Instale o que
combina com a sua plataforma — ou os dois, se você compartilha modelos entre o
backend Python e o frontend no navegador.

## Python (PyPI)

Requer **Python 3.11+**.

```bash
pip install ort-vision-sdk            # somente CPU (padrão)
pip install "ort-vision-sdk[gpu]"     # adiciona onnxruntime-gpu (CUDA / TensorRT)
pip install "ort-vision-sdk[opencv]"  # adiciona o backend de imagem OpenCV
pip install "ort-vision-sdk[dev]"     # ruff, mypy, pytest, build, twine
```

Dependências base: `onnxruntime>=1.17.0`, `numpy>=1.24.0`, `pillow>=10.0.0`.

### Extras

| Extra | Adiciona | Quando usar |
| --- | --- | --- |
| `gpu` | `onnxruntime-gpu` | Inferência em GPU NVIDIA via CUDA / TensorRT. |
| `opencv` | `opencv-python` | Backend de imagem OpenCV (alternativa ao Pillow). |
| `dev` | ruff, mypy, pytest, build, twine | Contribuir com o pacote. |

!!! warning "CPU vs. GPU"
    `onnxruntime` (CPU) e `onnxruntime-gpu` não devem coexistir no mesmo ambiente.
    Para usar GPU, instale o extra `gpu` em um ambiente limpo (sem o
    `onnxruntime` de CPU já presente), ou desinstale-o antes.

### Verificar a instalação

```bash
python -c "from ort_vision_sdk import Classifier, Detector, Segmenter; print('OK')"
```

## Web (npm)

```bash
npm install @mauriciobenjamin700/ort-vision-sdk-web onnxruntime-web
```

`onnxruntime-web` é uma **peer dependency** (faixa aceita: `>=1.17.0`). Você
escolhe a versão e distribui os arquivos `.wasm` correspondentes — o SDK não
empacota o runtime para que você controle a versão e o bundle.

!!! tip "Arquivos .wasm e WebGPU"
    Para que o WebGPU realmente seja usado (a ordem de providers padrão é
    `["webgpu", "wasm"]`), você precisa de um build recente do ORT-Web, um
    navegador Chromium com WebGPU habilitado e um contexto seguro (`https://` ou
    `localhost`). Sem isso, o runtime cai automaticamente para WebAssembly.

### Verificar a instalação

```bash
node -e "import('@mauriciobenjamin700/ort-vision-sdk-web').then(m => console.log(Object.keys(m)))"
```

## Próximos passos

- [Início rápido](inicio-rapido.md) — primeiros exemplos lado a lado.
- [Guia de classificação](guia/classificacao.md),
  [detecção](guia/deteccao.md) e [segmentação](guia/segmentacao.md).
