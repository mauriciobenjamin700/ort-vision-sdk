# Classificação

A tarefa `Classifier` aceita qualquer classificador ONNX com saída de formato
`(1, num_classes)` (estilo torchvision). Ela faz preprocessamento, normalização,
softmax opcional e resolução de rótulos por você, devolvendo um envelope
`ClassificationResults`.

## Construindo o classificador

=== "Python"

    ```python
    from ort_vision_sdk import Classifier

    clf = Classifier(
        "resnet50.onnx",
        labels="imagenet_labels.txt",   # ver "Rótulos" abaixo
        input_size=(224, 224),          # (largura, altura) padrão
        normalization="auto",           # lê os metadados do modelo e escolhe
        apply_softmax=True,             # False se o modelo já emite probabilidades
    )
    ```

    Parâmetros adicionais: `providers` (lista de execution providers) e
    `session_options` (uma `ort.SessionOptions`).

=== "Web (browser)"

    ```typescript
    import { Classifier } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const clf = await Classifier.create("/models/resnet50.onnx", {
      labels: ["tench", "goldfish", /* ... */],  // ou null + numClasses
      inputSize: [224, 224],                      // padrão
      normalization: "auto",                      // lê os metadados e escolhe
      applySoftmax: true,                         // padrão
      providers: ["webgpu", "wasm"],              // ordem padrão
    });
    ```

    Quando `labels` é `null`, informe `numClasses` para o SDK gerar
    `class_0`, `class_1`, ...

## Normalização: o modelo decide

O recorte chega ao classificador em `[0, 1]`. O que tem que acontecer depois
depende inteiramente de **como o seu modelo foi treinado** — e as duas famílias
mais comuns discordam:

| Família | Espera | Preset |
| --- | --- | --- |
| torchvision / timm | tensor normalizado com média e desvio do ImageNet | `"imagenet"` |
| Ultralytics (`YOLO(...).export()`) | `[0, 1]` cru, sem normalização nenhuma | `"ultralytics"` |

Por isso o padrão é `normalization="auto"`: o SDK **lê os metadados do próprio
modelo** e escolhe. Todo export do Ultralytics carrega `author: Ultralytics` e
`task: classify` no `.onnx` — o mesmo bloco de onde os nomes de classe já vinham.

!!! danger "O modo de falha que isto elimina"
    Até a 0.8.0 o padrão era ImageNet para todo mundo. Carregar um classificador
    Ultralytics com esse padrão o alimenta com um tensor que ele **nunca viu no
    treino** — e nada reclama: sem exceção, sem aviso, com a predição saindo na
    forma certa e simplesmente pior.

Para escolher à mão, ou fugir dos dois presets:

=== "Python"

    ```python
    Classifier("modelo.onnx", normalization="imagenet")
    Classifier("modelo.onnx", normalization="ultralytics")
    Classifier("modelo.onnx", normalization="none")
    Classifier("modelo.onnx", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    print(clf.normalization)   # "ultralytics" | "imagenet" | "none" | "custom"
    ```

=== "Web (browser)"

    ```typescript
    await Classifier.create(url, { normalization: "imagenet" });
    await Classifier.create(url, { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] });

    console.log(clf.normalization);
    ```

!!! tip "`mean` e `std` são independentes"
    Passar só um dos dois deixa o outro no valor do preset que o `auto`
    escolheria. Passar `mean`/`std` **junto** com `normalization` é erro
    (`ValueError` no Python, `RangeError` no web): são duas respostas para a
    mesma pergunta.

Pedir uma normalização não-identidade para um modelo Ultralytics ainda funciona,
mas emite aviso — o SDK não decide por você, só se recusa a ficar calado.

## Predizendo

`predict()` devolve uma lista de comprimento 1 — use `[0]`.

=== "Python"

    ```python
    r = clf.predict("dog.jpg")[0]

    print(r.cls, r.conf, r.name)   # top-1 (índice, confiança, rótulo)
    print(r.probs.top1)            # índice top-1
    print(r.probs.top5)            # array com os 5 índices mais prováveis
    print(r.probs.top1conf)        # confiança do top-1
    print(r.probs.top5conf)        # confianças do top-5
    print(r.probabilities[:5])     # tupla de dataclasses ClassProbability
    ```

=== "Web (browser)"

    ```typescript
    const r = (await clf.predict("/images/dog.jpg", { topK: 5 }))[0];

    console.log(r.cls, r.conf, r.name);   // top-1
    console.log(r.probs.top1, r.probs.top5);
    console.log(r.probs.top1conf, r.probs.top5conf);
    console.log(r.probabilities);          // ClassProbability[]
    ```

    `topK` controla quantas probabilidades por classe são materializadas em
    `probabilities`.

## A visão `Probs`

A visão em massa `probs` espelha a interface `Probs` do Ultralytics:
`top1`, `top5`, `top1conf`, `top5conf`, `data`. As dataclasses/objetos
`ClassProbability` carregam os campos verbosos (`class_id`/`classId`,
`class_name`/`className`, `probability`) e expõem aliases estilo Ultralytics
(`cls`, `name`).

## Rótulos

Veja a seção [Rótulos no guia Python](python.md#rotulos) (Python) e
[Rótulos no guia Web](web.md#rotulos) (Web) — as duas plataformas aceitam
preset (`"coco"`), lista/tupla, dict esparso, caminho de arquivo (Python) ou
`null` para auto-gerar.

## Veja também

- [Início rápido](../inicio-rapido.md)
- [Referência da API Python](../referencia/python.md)
- [Referência da API Web](../referencia/web.md)
