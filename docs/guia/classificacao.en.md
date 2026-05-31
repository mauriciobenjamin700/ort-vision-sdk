# Classification

The `Classifier` task accepts any ONNX classifier with an output shape of
`(1, num_classes)` (torchvision-style). It handles preprocessing, normalization,
optional softmax, and label resolution for you, returning a
`ClassificationResults` envelope.

## Building the classifier

=== "Python"

    ```python
    from ort_vision_sdk import Classifier

    clf = Classifier(
        "resnet50.onnx",
        labels="imagenet_labels.txt",   # see "Labels" below
        input_size=(224, 224),          # default (width, height)
        mean=(0.485, 0.456, 0.406),     # ImageNet mean by default
        std=(0.229, 0.224, 0.225),      # ImageNet std by default
        apply_softmax=True,             # False if your model already outputs probs
    )
    ```

    Additional parameters: `providers` (execution-provider list) and
    `session_options` (an `ort.SessionOptions`).

=== "Web (browser)"

    ```typescript
    import { Classifier } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const clf = await Classifier.create("/models/resnet50.onnx", {
      labels: ["tench", "goldfish", /* ... */],  // or null + numClasses
      inputSize: [224, 224],                      // default
      applySoftmax: true,                         // default
      providers: ["webgpu", "wasm"],              // default order
    });
    ```

    When `labels` is `null`, pass `numClasses` so the SDK auto-generates
    `class_0`, `class_1`, ...

## Predicting

`predict()` returns a length-1 list — use `[0]`.

=== "Python"

    ```python
    r = clf.predict("dog.jpg")[0]

    print(r.cls, r.conf, r.name)   # top-1 (index, confidence, label)
    print(r.probs.top1)            # top-1 index
    print(r.probs.top5)            # array of the 5 most-probable indices
    print(r.probs.top1conf)        # top-1 confidence
    print(r.probs.top5conf)        # top-5 confidences
    print(r.probabilities[:5])     # tuple of ClassProbability dataclasses
    ```

=== "Web (browser)"

    ```typescript
    const r = (await clf.predict("/images/dog.jpg", { topK: 5 }))[0];

    console.log(r.cls, r.conf, r.name);   // top-1
    console.log(r.probs.top1, r.probs.top5);
    console.log(r.probs.top1conf, r.probs.top5conf);
    console.log(r.probabilities);          // ClassProbability[]
    ```

    `topK` controls how many per-class probabilities are materialized in
    `probabilities`.

## The `Probs` view

The bulk `probs` view mirrors Ultralytics' `Probs` interface: `top1`, `top5`,
`top1conf`, `top5conf`, `data`. The `ClassProbability` dataclasses/objects carry
the verbose fields (`class_id`/`classId`, `class_name`/`className`,
`probability`) and expose Ultralytics-style aliases (`cls`, `name`).

## Labels

See [Labels in the Python guide](python.md#labels) (Python) and
[Labels in the Web guide](web.md#labels) (Web) — both platforms accept a preset
(`"coco"`), a list/tuple, a sparse dict, a file path (Python), or `null` to
auto-generate.

## See also

- [Quick start](../inicio-rapido.md)
- [Python API reference](../referencia/python.md)
- [Web API reference](../referencia/web.md)
