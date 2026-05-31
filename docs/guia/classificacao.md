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
        mean=(0.485, 0.456, 0.406),     # média ImageNet por padrão
        std=(0.229, 0.224, 0.225),      # desvio ImageNet por padrão
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
      applySoftmax: true,                         // padrão
      providers: ["webgpu", "wasm"],              // ordem padrão
    });
    ```

    Quando `labels` é `null`, informe `numClasses` para o SDK gerar
    `class_0`, `class_1`, ...

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
