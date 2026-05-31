# Guia Python

Detalhes específicos do pacote Python `ort-vision-sdk`: entradas aceitas,
resolução de rótulos, execution providers e as três variantes de inferência.

## Entradas aceitas

Todo `predict()` aceita o mesmo conjunto de entradas de imagem:

```python
from pathlib import Path
from PIL import Image
import numpy as np

clf.predict("dog.jpg")                              # caminho str
clf.predict(Path("dog.jpg"))                        # pathlib
clf.predict(open("dog.jpg", "rb").read())           # bytes crus (PNG, JPEG, ...)
clf.predict(Image.open("dog.jpg"))                  # PIL — convertido para RGB
clf.predict(np.zeros((480, 640, 3), dtype=np.uint8))  # ndarray HWC uint8 RGB
```

Para carregar uma imagem uma vez e reutilizá-la, use o mesmo loader interno:

```python
from ort_vision_sdk import load_image
img = load_image("dog.jpg")   # HWC uint8 RGB
clf.predict(img)
```

## Rótulos

As tarefas resolvem rótulos na construção via `resolve_labels`:

```python
from ort_vision_sdk import Classifier, Detector, COCO_CLASSES, resolve_labels

# 1) Preset embutido (atualmente: "coco")
det = Detector("yolov8n.onnx", labels="coco")

# 2) Lista / tupla explícita
clf = Classifier("model.onnx", labels=["cat", "dog", "fox"])

# 3) Dict esparso — lacunas preenchidas com "class_<id>"
clf = Classifier("model.onnx", labels={0: "cat", 2: "fox"})

# 4) Caminho de arquivo — uma classe por linha
clf = Classifier("model.onnx", labels="imagenet_labels.txt")

# 5) None — auto-gera "class_0", "class_1", ... (só quando o formato de saída
#    do modelo é estaticamente conhecido)
clf = Classifier("model.onnx", labels=None)
```

`names` em todo resultado é o mapeamento canônico `dict[int, str]` (espelha o
`model.names` do Ultralytics).

## Execution providers

Por padrão o SDK escolhe o primeiro provider disponível na ordem de preferência
do ORT. Para fixar um backend específico, passe `providers=` com aliases curtos
ou nomes canônicos do ORT:

```python
det = Detector("yolov8n.onnx", providers=["cuda", "cpu"])
det = Detector("yolov8n.onnx", providers=["tensorrt", "cuda", "cpu"])
det = Detector("yolov8n.onnx", providers=["CUDAExecutionProvider"])  # nome canônico
```

Aliases suportados: `"cpu"`, `"cuda"`, `"tensorrt"`, `"directml"`, `"coreml"`,
`"openvino"`, `"rocm"`. Qualquer outra coisa é repassada literalmente ao ORT.

Para controle fino (otimização de grafo, threading, profiling), passe uma
`ort.SessionOptions`:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4

det = Detector("yolov8n.onnx", session_options=opts)
```

## Inferência assíncrona

Cada classe expõe duas variantes async de `predict()` com a mesma assinatura da
síncrona. Escolha a que combina com o seu perfil de concorrência:

| Método | Mecanismo | Use quando |
| --- | --- | --- |
| `predict()` | Síncrono | Scripts, notebooks, pipelines de batch sem event loop. |
| `async_predict()` | `asyncio.to_thread` | Caminho async padrão — handlers FastAPI/AnyIO/Quart. Off-loada o pipeline inteiro (pré + run + pós) para o thread pool do executor padrão do asyncio, liberando o event loop. Uma thread Python por inferência em voo. |
| `ort_async_predict()` | `InferenceSession.run_async` | Alta concorrência — muitos `await` simultâneos compartilham um único thread pool. Pré/pós-processamento rodam na thread do event loop; o run do modelo é despachado para o pool interno do ONNX Runtime configurado via `SessionOptions`. Requer `onnxruntime>=1.16`. |

A mesma divisão existe na sessão subjacente — `OrtSession.async_run` /
`OrtSession.ort_async_run` — para quem constrói o próprio pipeline.

### Handler FastAPI (async padrão)

```python
from fastapi import FastAPI, UploadFile
from ort_vision_sdk import Detector

app = FastAPI()
det = Detector("yolov8n.onnx")

@app.post("/detect")
async def detect(file: UploadFile) -> dict[str, list[dict[str, float | int | str]]]:
    image_bytes = await file.read()
    result = (await det.async_predict(image_bytes))[0]
    return {
        "detections": [
            {"name": d.name, "conf": d.conf, "x1": d.box.x1, "y1": d.box.y1,
             "x2": d.box.x2, "y2": d.box.y2}
            for d in result
        ]
    }
```

### Batch de alta concorrência (pool do ORT)

```python
import asyncio
import onnxruntime as ort
from ort_vision_sdk import Detector

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.inter_op_num_threads = 1

det = Detector("yolov8n.onnx", session_options=opts)

async def detect_all(paths: list[str]) -> list[list]:
    return await asyncio.gather(*(det.ort_async_predict(p) for p in paths))

results = asyncio.run(detect_all([f"img_{i}.jpg" for i in range(200)]))
```

### Regra de bolso

- **Chamada async pontual num handler de requisição** → `async_predict`.
- **Centenas de inferências concorrentes** (worker de fila, endpoint de batch)
  → `ort_async_predict`.

## Veja também

- [Referência da API Python](../referencia/python.md)
- [Guia Web](web.md) — o equivalente no navegador.
