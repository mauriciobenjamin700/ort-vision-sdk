# The model decides

Two things about an `.onnx` live **inside the file**: the resolution it accepts
as input, and the class names it emits. Restating those numbers in configuration
is where the classic headache comes from — the configuration ages, the file gets
re-exported, and the two drift apart with nobody noticing.

The SDK reads both from the model. This page shows how, and what is left for you
to configure. 🚀

## The problem: 640 against 224

An Ultralytics detection export comes out at 640×640. A classification export
(`-cls`) comes out at **224×224** — that is the tool's default. If your code
assumes 640 for both, the classifier dies mid-inference:

```text
Got invalid dimensions for input: images for the following indices
index: 2 Got: 640 Expected: 224
```

!!! danger "There is no way to see that coming from the outside"
    The number lives in the graph. No configuration, manifest or constant
    sitting next to the file is the source of truth about it — only the `.onnx`
    itself is.

## The fix: ask the graph

You configure nothing:

```python
from ort_vision_sdk import Classifier

clf = Classifier("classify.onnx")
print(clf.input_size)
#> (224, 224)
```

The SDK read the shape the graph declares (`[1, 3, 224, 224]`) and will
preprocess at that resolution. A detector in the same program resolves its own:

```python
from ort_vision_sdk import Detector

det = Detector("detect.onnx")
print(det.input_size)
#> (640, 640)
```

!!! tip "Read back what actually ran"
    `task.input_size` is the resolution inference **really** used — handy for
    logs and telemetry, where reporting the configured value hides exactly the
    bug you are hunting.

### What if I pass `input_size`?

The graph wins, and the SDK says so:

```python
clf = Classifier("classify.onnx", input_size=(640, 640))
#> UserWarning: The model declares a 224x224 input; ignoring the requested
#>              640x640, which ONNX Runtime would reject.
print(clf.input_size)
#> (224, 224)
```

That is deliberate. A static shape is not a preference: it is the only thing
ONNX Runtime will accept. Obeying you there would just trade a fixable problem
for a failed run.

### When `input_size` still matters

A model exported with dynamic axes (`dynamic=True` in Ultralytics) declares
height and width as symbols, so it accepts many resolutions. The graph then has
nothing to say, and your value stands:

```python
clf = Classifier("dynamic.onnx", input_size=(384, 384))
print(clf.input_size)
#> (384, 384)
```

!!! info "Precedence, in one line"
    **static graph → what you passed → the task default** (224 for
    classification, 640 for detection/segmentation).

## Labels straight from the model

Ultralytics writes `names` into the `.onnx` metadata — the `dict[int, str]` map
from class id to name. A hand-maintained list beside it can be reordered by
accident, and the effect is the worst kind: nothing fails, predictions simply
swap classes.

With no `labels`, the SDK uses what the model declares:

```python
from ort_vision_sdk import Detector

det = Detector("detect.onnx")
print(det.labels)
#> ('ocular-mucosa',)
```

!!! check "This also fixes an old stumble"
    A custom detector used to **fail** without an explicit `labels`: the default
    was the 80-name COCO preset, which disagreed with the model's class count.
    Now it resolves its own name.

Precedence follows the same idea:

```python
# 1) What you pass always wins
det = Detector("detect.onnx", labels=["mucosa"])

# 2) No labels: the model's `names`
det = Detector("detect.onnx")

# 3) Model without `names`: the COCO preset (detection/segmentation)
#    or "class_0", "class_1", ... (classification)
```

!!! note "Safe parsing"
    The value is read with `ast.literal_eval`, so malformed — or hostile —
    metadata is **rejected**, never executed. A map that is not a `dict` of
    contiguous integers starting at zero is discarded whole, rather than applied
    halfway.

## Reading the metadata yourself

The full map is available on the session:

```python
from ort_vision_sdk import Classifier

clf = Classifier("classify.onnx")
print(clf.session.metadata["task"])
#> classify
print(clf.session.input_shape)
#> (1, 3, 224, 224)
```

And the pure helpers behind all of this are public, for anyone assembling their
own pipeline:

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

??? note "Technical details: backends that cannot read metadata"
    A backend that only forwards tensors to a native runtime (an Android bridge,
    `onnxruntime-web` through Pyodide) cannot read the metadata map. That is why
    the capability is a separate protocol, `MetadataBackend`, instead of a
    required member of `InferenceBackend`: tasks probe with `read_metadata()`
    and simply get nothing when the backend does not offer it. Backends written
    before this keep working.

## Recap

- The input resolution comes from the **graph**; `input_size` is a fallback for
  dynamic-axis models.
- Passing a size that contradicts a static graph raises a `UserWarning` and is
  ignored — ORT would reject it anyway.
- `task.input_size` tells you what inference actually used.
- With no `labels`, names come from the model metadata (Ultralytics' `names`),
  falling back to COCO (detection/segmentation) or `class_<id>`
  (classification).
- `session.metadata`, `session.input_shape` and the `spatial_input_size` /
  `resolve_input_size` / `model_names` helpers are public.

## See also

- [Python guide](python.md) — accepted inputs, providers, async inference.
- [Web (browser)](web.md) — the same behavior in the TypeScript SDK.
- [Inference backends](backends.md) — the `InferenceBackend` protocol.
