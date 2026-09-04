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
pip install "ort-vision-sdk[compose]" # adiciona onnx, para fundir modelos num pipeline
pip install "ort-vision-sdk[dev]"     # ruff, mypy, pytest, build, twine
```

Dependências base: `onnxruntime>=1.17.0`, `numpy>=1.24.0`, `pillow>=10.0.0`.

### Extras

| Extra | Adiciona | Quando usar |
| --- | --- | --- |
| `gpu` | `onnxruntime-gpu` | Inferência em GPU NVIDIA via CUDA / TensorRT. |
| `opencv` | `opencv-python` | Backend de imagem OpenCV (alternativa ao Pillow). |
| `compose` | `onnx` | Fundir detector + classificador num único `.onnx` — ver [Pipelines fundidos](guia/pipeline.md). |
| `dev` | ruff, mypy, pytest, build, twine | Contribuir com o pacote. |

!!! info "`compose` é só para o build"
    O `onnx` (a biblioteca que reescreve o protobuf) é necessário apenas para
    **fundir** modelos, um passo que você roda uma vez junto do seu pipeline de
    export. **Rodar** o `.onnx` fundido não precisa de nada além do
    `onnxruntime` que já vem na instalação base.

!!! warning "CPU vs. GPU"
    `onnxruntime` (CPU) e `onnxruntime-gpu` não devem coexistir no mesmo ambiente.
    Para usar GPU, instale o extra `gpu` em um ambiente limpo (sem o
    `onnxruntime` de CPU já presente), ou desinstale-o antes.

!!! danger "GPU disponível ≠ GPU carregável"
    `onnxruntime.get_available_providers()` responde **"isto foi compilado no
    wheel"**, não "isto consegue carregar". O `onnxruntime-gpu` lista
    `CUDAExecutionProvider` sempre, e ainda assim registra CPU quando o loader
    dinâmico não acha o `libcudnn.so.9`. O resultado é um deploy que pediu GPU,
    recebeu CPU sem erro nenhum, e só aparece na conta de latência semanas
    depois.

    O caso é mais escorregadio do que parece: **importar `torch` antes** faz o
    CUDA carregar, porque o wheel do torch traz o cuDNN e o carrega no processo.
    O mesmo código funciona ou não dependendo da ordem de import de uma
    biblioteca que o SDK nem depende.

    A partir da 0.9.0 o SDK reconcilia isso: `session.providers` lê de volta o
    que o ORT **registrou**, e pedir um provider por nome e não recebê-lo emite
    um `UserWarning` em vez de silêncio.

### Verificar a instalação

```bash
python -c "from ort_vision_sdk import Classifier, Detector, Segmenter; print('OK')"
```

E, quando a intenção é rodar em GPU, confirme **onde** ela de fato rodou:

```python
from ort_vision_sdk import OrtSession

session = OrtSession("yolov8n.onnx", providers=["cuda"])

print(session.requested_providers)  # ['CUDAExecutionProvider'] — o que foi pedido
print(session.providers)            # o que o ORT registrou de verdade
```

!!! tip "Vale para as tasks também"
    `Detector`, `Classifier` e `Segmenter` constroem um `OrtSession` por baixo e
    o expõem em `.session`, então `detector.session.providers` responde a mesma
    pergunta quando você não injetou um backend próprio.

Se a segunda linha imprimir apenas `['CPUExecutionProvider']`, o cuDNN não está
alcançável — instale-o, ou aponte o `LD_LIBRARY_PATH` para onde ele está.

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

E, no navegador, confirme **onde** a inferência de fato rodou:

```typescript
const det = await Detector.create("/models/yolov8n.onnx", {
  providers: ["webgpu"],
});

console.log(det.session.requestedProviders);  // ["webgpu"] — o que foi pedido
console.log(det.session.providers);           // o que este navegador pode oferecer
```

!!! danger "WebGPU disponível ≠ WebGPU carregável"
    O ORT-Web **não expõe** equivalente ao `getProviders()` do Node: não dá para
    perguntar em qual provider a sessão acabou. Uma página que pede `webgpu` num
    aparelho sem ele roda em WASM várias vezes mais devagar, sem erro nenhum.

    A partir da 0.8.0 o SDK estreita a lista pedida pelo que o navegador
    realmente consegue oferecer (`navigator.gpu` existir **e** devolver adapter),
    e pedir um provider por nome e não conseguir emite `console.warn`. É
    best-effort: um provider que sobrevive à checagem ainda pode falhar dentro do
    ORT por um motivo que o navegador não expõe.

## Próximos passos

- [Início rápido](inicio-rapido.md) — primeiros exemplos lado a lado.
- [Guia de classificação](guia/classificacao.md),
  [detecção](guia/deteccao.md) e [segmentação](guia/segmentacao.md).
- [Pipelines fundidos](guia/pipeline.md) — dois modelos, um arquivo, uma sessão.
