"""Build-time model composition: fuse several models into one ONNX graph.

This subpackage is the only part of the SDK that needs the ``onnx`` library,
because it is the only part that rewrites model files rather than running them.
Install it with the ``compose`` extra::

    pip install "ort-vision-sdk[compose]"

Composing is a build step you run once, next to your export pipeline. The
result is a plain ``.onnx`` file, so *running* it needs nothing beyond the
``onnxruntime`` the SDK already depends on — including in the browser, where
``@mauriciobenjamin700/ort-vision-sdk-web`` loads the very same file.

Example:
    >>> from ort_vision_sdk.compose import fuse_detect_classify
    >>> fuse_detect_classify("yolov8n.onnx", "resnet18.onnx", "pipeline.onnx")

The configuration chosen here is recorded inside the fused file, so the runtime
side never restates it — see :mod:`ort_vision_sdk.fusion` for that contract and
:class:`~ort_vision_sdk.tasks.pipeline.DetectClassify` for the runtime itself.
"""

from ort_vision_sdk.compose.bridge import MIN_OPSET, BridgeGraph, build_bridge
from ort_vision_sdk.compose.detect_classify import Normalization, fuse_detect_classify

__all__: list[str] = [
    "MIN_OPSET",
    "BridgeGraph",
    "Normalization",
    "build_bridge",
    "fuse_detect_classify",
]
