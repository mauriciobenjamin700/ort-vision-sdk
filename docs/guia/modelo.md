# O modelo manda

Duas coisas sobre um `.onnx` estão **dentro do arquivo**: a resolução que ele
aceita na entrada e os nomes das classes que ele emite. Repetir esses números na
configuração é o que gera a dor de cabeça clássica — a configuração envelhece, o
arquivo é re-exportado, e os dois divergem sem ninguém perceber.

O SDK lê ambos do modelo. Esta página mostra como, e o que sobra para você
configurar. 🚀

## O problema: 640 contra 224

Um export de detecção do Ultralytics sai em 640×640. Um export de
classificação (`-cls`) sai em **224×224** — é o default da ferramenta. Se o seu
código assume 640 para os dois, o classificador morre no meio da inferência:

```text
Got invalid dimensions for input: images for the following indices
index: 2 Got: 640 Expected: 224
```

!!! danger "Esse erro não tem como ser previsto de fora"
    O número vive no grafo. Nenhuma configuração, manifest ou constante ao lado
    do arquivo é fonte da verdade sobre ele — só o próprio `.onnx` é.

## A solução: perguntar ao grafo

Você não configura nada:

```python
from ort_vision_sdk import Classifier

clf = Classifier("classify.onnx")
print(clf.input_size)
#> (224, 224)
```

O SDK leu o shape declarado pelo grafo (`[1, 3, 224, 224]`) e vai
pré-processar nessa resolução. Um detector no mesmo programa resolve o dele:

```python
from ort_vision_sdk import Detector

det = Detector("detect.onnx")
print(det.input_size)
#> (640, 640)
```

!!! tip "Leia de volta o que rodou"
    `task.input_size` é a resolução que a inferência **realmente** usou — útil
    para logs e telemetria, onde reportar o valor configurado esconde
    exatamente o bug que você está caçando.

### E se eu passar `input_size`?

O grafo ganha, e o SDK avisa:

```python
clf = Classifier("classify.onnx", input_size=(640, 640))
#> UserWarning: The model declares a 224x224 input; ignoring the requested
#>              640x640, which ONNX Runtime would reject.
print(clf.input_size)
#> (224, 224)
```

Isso é de propósito. Um shape estático não é uma preferência: é a única coisa
que o ONNX Runtime aceita. Obedecer você ali só trocaria um problema
corrigível por uma execução que falha.

### Quando `input_size` ainda importa

Um modelo exportado com eixos dinâmicos (`dynamic=True` no Ultralytics) declara
altura e largura como símbolos, e aí ele aceita várias resoluções. Nesse caso o
grafo não tem o que dizer, e o seu valor vale:

```python
clf = Classifier("dynamic.onnx", input_size=(384, 384))
print(clf.input_size)
#> (384, 384)
```

!!! info "Precedência, em uma linha"
    **grafo estático → o que você passou → default da tarefa** (224 para
    classificação, 640 para detecção/segmentação).

## Rótulos que vêm do próprio modelo

O Ultralytics grava `names` nos metadados do `.onnx` — o mapa `dict[int, str]`
de id de classe para nome. Uma lista mantida à mão do lado pode ser reordenada
por acidente, e o efeito é o pior possível: nada falha, as predições
simplesmente trocam de classe.

Sem `labels`, o SDK usa o que o modelo declara:

```python
from ort_vision_sdk import Detector

det = Detector("detect.onnx")
print(det.labels)
#> ('ocular-mucosa',)
```

!!! check "Isso também conserta um tropeço antigo"
    Antes, um detector custom **falhava** sem `labels` explícito: o default era
    o preset COCO de 80 nomes, que discordava da contagem de classes do modelo.
    Agora ele resolve o próprio nome.

A precedência segue a mesma ideia:

```python
# 1) O que você passa sempre ganha
det = Detector("detect.onnx", labels=["mucosa"])

# 2) Sem labels: os `names` do modelo
det = Detector("detect.onnx")

# 3) Modelo sem `names`: preset COCO (detecção/segmentação)
#    ou "class_0", "class_1", ... (classificação)
```

!!! note "Parsing seguro"
    O valor é lido com `ast.literal_eval`, então um metadado malformado — ou
    hostil — é **rejeitado**, nunca executado. Um mapa que não seja um `dict`
    de inteiros contíguos começando em zero é descartado inteiro, em vez de
    aplicado pela metade.

## Lendo os metadados você mesmo

O mapa completo está disponível na sessão:

```python
from ort_vision_sdk import Classifier

clf = Classifier("classify.onnx")
print(clf.session.metadata["task"])
#> classify
print(clf.session.input_shape)
#> (1, 3, 224, 224)
```

E os helpers puros por trás de tudo isso são públicos, para quem monta o próprio
pipeline:

```python
from ort_vision_sdk import model_names, resolve_input_size, spatial_input_size

spatial_input_size((1, 3, 224, 224))
#> (224, 224)
spatial_input_size((1, 3, "h", "w"))
#> None
resolve_input_size(graph_shape=(1, 3, 224, 224), requested=None, fallback=(640, 640))
#> (224, 224)
model_names({"names": "{0: 'deworm', 1: 'not_deworm'}"})
#> {0: 'deworm', 1: 'not_deworm'}
```

??? note "Detalhes técnicos: backends que não leem metadados"
    Um backend que só repassa tensores para um runtime nativo (bridge Android,
    `onnxruntime-web` via Pyodide) não consegue ler o mapa de metadados. Por
    isso a capacidade é um protocolo separado, `MetadataBackend`, e não um
    membro obrigatório de `InferenceBackend`: as tarefas consultam com
    `read_metadata()` e simplesmente não recebem nada quando o backend não
    oferece. Backends escritos antes disso continuam válidos.

## Recapitulando

- A resolução de entrada vem do **grafo**; `input_size` é fallback para modelos
  de eixo dinâmico.
- Passar um tamanho que contradiz um grafo estático emite `UserWarning` e é
  ignorado — o ORT rejeitaria mesmo.
- `task.input_size` diz o que a inferência usou de verdade.
- Sem `labels`, os nomes vêm dos metadados do modelo (`names` do Ultralytics),
  caindo para COCO (detecção/segmentação) ou `class_<id>` (classificação).
- `session.metadata`, `session.input_shape` e os helpers `spatial_input_size` /
  `resolve_input_size` / `model_names` estão públicos.

## Veja também

- [Guia Python](python.md) — entradas aceitas, providers, inferência assíncrona.
- [Web (browser)](web.md) — o mesmo comportamento no SDK TypeScript.
- [Backends de inferência](backends.md) — o protocolo `InferenceBackend`.
