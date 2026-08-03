# Medindo o custo da inferência

Você tem o modelo rodando. A próxima pergunta é sempre a mesma: **por que
demorou isso?** 🐌

Um `predict()` de 2 segundos pode ser um forward pass pesado — ou pode ser um
download de 12 MB que aconteceu uma vez e nunca mais. As duas situações pedem
ações opostas, e sem número você não sabe qual das duas está vendo.

Todo envelope de resultado carrega um `speed` com o tempo de cada etapa.

## O básico

```python
from ort_vision_sdk import Detector

det = Detector("yolov8n.onnx", labels="coco")
results = det.predict("street.jpg")

print(results[0].speed)
# {"load": 84.2, "preprocess": 11.7, "inference": 118.9, "postprocess": 6.4}
```

Valores em **milissegundos**. Mesma coisa no browser:

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

const det = await Detector.create("/models/yolov8n.onnx");
const results = await det.predict("/images/street.jpg");

console.log(results[0].speed);
// { load: 84.2, preprocess: 11.7, inference: 118.9, postprocess: 6.4 }
```

!!! tip "Mesmo formato nos dois SDKs"
    Python devolve um `dict[str, float]`, TypeScript devolve um objeto
    `Speed`. As chaves e as fronteiras de medição são idênticas — um
    dashboard consegue tratar os dois do mesmo jeito.

## O que cada etapa significa

| Chave | Cobre |
| --- | --- |
| `load` | Ler/baixar a entrada e decodificar para `RGBImage`/`ndarray` |
| `preprocess` | Letterbox/resize, normalização e empacotamento do tensor |
| `inference` | O forward pass no ONNX Runtime |
| `postprocess` | Decodificar a saída — NMS, montagem de máscara, top-k |

`preprocess`, `inference` e `postprocess` são exatamente as três chaves que o
Ultralytics reporta, medidas nas mesmas fronteiras. `load` é nosso: o
`predict()` aceita caminho, bytes ou URL e faz a decodificação por dentro. Se
esse custo entrasse em `preprocess`, a leitura ficaria mentirosa — numa
primeira chamada com cache frio, `load` costuma ser a maior fatia de todas.

!!! warning "As etapas se somam ao total, mas não ao seu relógio"
    As quatro etapas ladrilham o `predict()` inteiro, sem buracos. O que elas
    **não** incluem é o `Detector(...)`/`Detector.create(...)` — carregar o
    modelo e subir a sessão ORT acontece antes, uma única vez, e não aparece
    no `speed`.

## Lendo o resultado

```python
speed = results[0].speed
total = sum(speed.values())

for stage, ms in speed.items():
    print(f"{stage:12} {ms:7.1f} ms  {ms / total:5.1%}")
```

```text
load            84.2 ms  38.5%
preprocess      11.7 ms   5.3%
inference      118.9 ms  54.3%
postprocess      6.4 ms   2.9%
```

Três leituras que aparecem na prática:

- **`load` domina** — está buscando a imagem pela rede a cada chamada.
  Decodifique uma vez e passe o array/`RGBImage` já pronto.
- **`inference` domina** — é o modelo mesmo. Quantize, reduza o `inputSize`,
  ou confira se o provider que você pediu foi de fato o que o ORT resolveu
  (`session.providers`).
- **`postprocess` domina** — quase sempre NMS com `conf_threshold` baixo
  demais deixando milhares de candidatos passarem.

## Medindo o seu pipeline junto

Seu app raramente é só `predict()`. Se você recorta uma ROI entre uma detecção
e uma classificação, esse recorte também custa — e o `speed` de cada
`predict()` não enxerga o que acontece entre eles.

O `SpeedTimer` é a mesma peça que os tasks usam por dentro, exportada para
você medir com as mesmas fronteiras:

```python
from ort_vision_sdk.core import SpeedTimer

timer = SpeedTimer()
image = load_my_image(path)
timer.stage("load")

roi = detector.predict(image)[0]
timer.stage("inference")

crop = crop_to_box(image, roi.boxes.xyxy[0])
timer.stage("postprocess")

print(timer.speed())
```

```typescript
import { SpeedTimer } from "@mauriciobenjamin700/ort-vision-sdk-web";

const timer = new SpeedTimer();
const image = await loadMyImage(url);
timer.stage("load");

const roi = (await detector.predict(image))[0];
timer.stage("inference");

const crop = cropToBox(image, roi.boxes.xyxy);
timer.stage("postprocess");

console.log(timer.speed());
```

Cada `stage()` fecha o intervalo anterior e credita o tempo ao nome dado —
sem pares de start/stop para esquecer. Chamar o mesmo nome duas vezes
**acumula**, então dá para somar duas passadas de inferência na mesma chave.

## Recapitulando

- `results[0].speed` traz `load`, `preprocess`, `inference` e `postprocess`
  em milissegundos, em Python e no browser. ✅
- As três chaves do Ultralytics medem as mesmas fronteiras; `load` é nossa,
  porque o `predict()` decodifica a entrada por dentro.
- Carregar o modelo **não** está no `speed` — é custo de inicialização.
- `SpeedTimer` mede as etapas do seu próprio pipeline com as mesmas regras.
