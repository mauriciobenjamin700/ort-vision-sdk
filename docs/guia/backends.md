# Backends de inferência

Por padrão, o `ort-vision-sdk` roda a inferência com o **ONNX Runtime**
in-process (a classe `OrtSession`). Mas todo o resto do SDK — pré-processamento,
pós-processamento, decode de resultado — é **NumPy puro**. A partir da
**v0.4.0** você pode trocar só o motor de inferência por um **backend** seu,
mantendo o pipeline em Python.

!!! tip "Por que isso importa"
    Em alguns ambientes **não existe wheel Python do `onnxruntime`**:

    - no **navegador** (Pyodide/WASM) a inferência roda no `onnxruntime-web` (JS);
    - no **Android** ela roda na AAR nativa `onnxruntime-android`.

    Em ambos, o SDK roda normalmente em Python e só a chamada de inferência
    cruza a ponte para o runtime nativo. 🚀

## O protocolo `InferenceBackend`

Um backend é qualquer objeto que satisfaça o protocolo
`InferenceBackend` — expõe a metadata do modelo e roda a inferência sobre um
dicionário `{nome_da_entrada: ndarray}`:

```python
from typing import Protocol
import numpy as np

class InferenceBackend(Protocol):
    @property
    def input_name(self) -> str: ...
    @property
    def input_shape(self) -> tuple[int | str, ...]: ...
    @property
    def output_names(self) -> list[str]: ...
    @property
    def output_shapes(self) -> list[tuple[int | str, ...]]: ...

    def run(self, feeds: dict[str, np.ndarray], *,
            output_names: list[str] | None = None) -> list[np.ndarray]: ...
    async def async_run(self, feeds, *, output_names=None) -> list[np.ndarray]: ...
    async def ort_async_run(self, feeds, *, output_names=None) -> list[np.ndarray]: ...
```

!!! note
    O backend padrão, `OrtSession`, já satisfaz esse protocolo. Você só implementa
    um quando quer rodar fora do ONNX Runtime in-process.

## Exemplo completo: um backend de eco

Um backend mínimo, executável, que devolve uma saída fixa (útil para testes):

```python
import numpy as np
from ort_vision_sdk import Detector

class EchoBackend:
    """Backend de teste — devolve sempre a mesma saída YOLO (sem detecções)."""

    def __init__(self) -> None:
        self._outputs = [np.zeros((1, 84, 8400), dtype=np.float32)]  # 4 + 80 classes

    @property
    def input_name(self) -> str:
        return "images"

    @property
    def input_shape(self) -> tuple[int | str, ...]:
        return (1, 3, 640, 640)

    @property
    def input_names(self) -> list[str]:
        return ["images"]

    @property
    def input_shapes(self) -> list[tuple[int | str, ...]]:
        return [(1, 3, 640, 640)]

    @property
    def output_names(self) -> list[str]:
        return ["output0"]

    @property
    def output_shapes(self) -> list[tuple[int | str, ...]]:
        return [(1, 84, 8400)]

    def run(self, feeds, *, output_names=None):
        return self._outputs

    async def async_run(self, feeds, *, output_names=None):
        return self.run(feeds, output_names=output_names)

    async def ort_async_run(self, feeds, *, output_names=None):
        return self.run(feeds, output_names=output_names)


# Injete o backend — o `model_path` é ignorado (o backend é dono do carregamento).
det = Detector("unused.onnx", backend=EchoBackend())
results = det.predict(np.zeros((480, 640, 3), dtype=np.uint8))
print(len(results[0]))   # 0 detecções (saída zerada)
```

O pré/pós (letterbox, normalização, NMS, parsing) rodou em Python; só o `run`
passou pelo backend.

!!! info "Bridge real (Android/web)"
    Num backend de ponte, o `run` serializa o `ndarray` de entrada, envia ao
    runtime nativo (AAR / `onnxruntime-web`) e desserializa a saída de volta —
    a metadata (`input_name`, `output_shapes`, ...) vem do modelo carregado do
    outro lado da ponte.

## Injetando em qualquer tarefa

`Detector`, `Classifier` e `Segmenter` aceitam o mesmo argumento:

```python
Classifier("m.onnx", backend=meu_backend)
Detector("m.onnx", backend=meu_backend)
Segmenter("m.onnx", backend=meu_backend)
```

!!! warning
    Quando você passa `backend=`, os argumentos `model_path`, `providers` e
    `session_options` são **ignorados** — o backend é responsável por carregar o
    modelo e escolher o acelerador.

## Recap

- O SDK separa **pipeline** (NumPy, sempre em Python) de **motor de inferência**
  (o backend).
- Implemente `InferenceBackend` para rodar onde não há wheel do `onnxruntime`
  (navegador, Android) ou para plugar outro runtime.
- Injete via `backend=` em qualquer tarefa; o padrão continua sendo o
  `OrtSession` (ONNX Runtime in-process), **100% retrocompatível**.
