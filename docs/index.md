# ort-vision-sdk

SDKs de alto nível para **inferência de visão computacional** sobre o
[ONNX Runtime](https://onnxruntime.ai/). O repositório distribui dois pacotes
irmãos — a mesma API orientada a tarefas (`Classifier`, `Detector`, `Segmenter`)
e os mesmos formatos de resultado tipados — um para Python (servidores/scripts)
e um para o navegador (TypeScript).

| Pacote | Registro | Diretório | Instalação |
| --- | --- | --- | --- |
| `ort-vision-sdk` | PyPI | [`sdk-python/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-python) | `pip install ort-vision-sdk` |
| `@mauriciobenjamin700/ort-vision-sdk-web` | npm | [`sdk-js-web/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-js-web) | `npm install @mauriciobenjamin700/ort-vision-sdk-web onnxruntime-web` |

!!! info "Idioma / Language"
    Esta documentação é bilíngue. Use o seletor de idioma no topo da página para
    alternar entre **Português (BR)** e **English (US)**.

## O que é

Usar o `onnxruntime` diretamente obriga você a escolher e configurar execution
providers, fazer letterbox/resize/normalização/`to_chw`/batch da imagem, decodificar
a saída do modelo (grids de âncoras, NMS, protótipos de máscara), mapear caixas de
volta da entrada com letterbox para a imagem original e resolver índices de classe
para rótulos legíveis — tudo repetido por família de tarefa.

O `ort-vision-sdk` faz tudo isso por você e devolve um resultado **tipado**, no
formato [Ultralytics](https://docs.ultralytics.com/) (`boxes.xyxy`, `cls`, `conf`,
`names`, ...), para que código existente migre com edições mínimas. De uma imagem
crua (path, bytes, array NumPy ou imagem PIL no Python; URL/Blob/canvas/etc. no
navegador) até um resultado tipado em **uma chamada**.

## O que vem na caixa

| Tarefa | Classe | Modelos suportados |
| --- | --- | --- |
| Classificação | `Classifier` | Qualquer classificador ONNX com saída `(1, num_classes)` (estilo torchvision) |
| Detecção de objetos | `Detector` | Cabeças YOLO anchor-free: v8, v9, v10, v11, v12, v26 |
| Segmentação de instância | `Segmenter` | Cabeças YOLO-seg: v8-seg, v11-seg, v26-seg (+ protótipos) |

As três tarefas retornam o mesmo formato de envelope — uma lista (`list[Results]`
no Python, `Results[]` no Web) de comprimento 1 por imagem — então você troca de
tarefa sem reescrever o código que consome o resultado.

## Instalação rápida

=== "Python"

    ```bash
    pip install ort-vision-sdk            # somente CPU (padrão)
    pip install "ort-vision-sdk[gpu]"     # adiciona onnxruntime-gpu (CUDA / TensorRT)
    pip install "ort-vision-sdk[opencv]"  # adiciona o backend de imagem OpenCV
    ```

    Requer Python **3.10+**.

=== "Web (browser)"

    ```bash
    npm install @mauriciobenjamin700/ort-vision-sdk-web onnxruntime-web
    ```

    `onnxruntime-web` é uma **peer dependency** — você traz sua própria versão e
    distribui os arquivos `.wasm` correspondentes.

## Primeiros passos

=== "Python"

    ```python
    from ort_vision_sdk import Detector

    det = Detector("yolov8n.onnx")          # labels="coco" por padrão
    result = det.predict("street.jpg")[0]   # list[DetectionResults], comprimento 1
    for d in result:
        print(d.name, d.conf, d.box.xyxy)
    ```

=== "Web (browser)"

    ```typescript
    import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const det = await Detector.create("/models/yolov8n.onnx");
    const result = (await det.predict("/images/street.jpg"))[0];
    for (const d of result) {
      console.log(d.className, d.confidence, d.bbox.asXyxy());
    }
    ```

Continue em [Instalação](instalacao.md) e [Início rápido](inicio-rapido.md).

## Status

Alpha — a API pública é estável o suficiente para construir em cima, mas versões
menores podem introduzir mudanças incompatíveis até a 1.0. Fixe a faixa de versão
contra a qual você desenvolve.

- Código-fonte & issues: <https://github.com/mauriciobenjamin700/ort-vision-sdk>
- Pacote Python: <https://pypi.org/project/ort-vision-sdk/>
- Pacote Web: <https://www.npmjs.com/package/@mauriciobenjamin700/ort-vision-sdk-web>

## Licença

[MIT](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/LICENSE).
