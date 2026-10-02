"""Build the model variants the browser benchmark compares, from any export.

``bench/browser/run.html`` measures one model four ways — as exported,
pre-optimized, INT8 and FP16 — and reads them from ``bench/browser/models/``,
which git ignores: the models are yours, not the repository's. This script
fills that directory from a single export and a few images:

.. code-block:: bash

    PYTHONPATH=sdk-python/src sdk-python/.venv/bin/python scripts/prepare_browser_bench.py \\
        --model path/to/yolo11n-seg.onnx --image path/to/frame.jpg \\
        --calibration path/to/calibration_images/

It writes ``model.onnx`` (a copy), ``model.opt.onnx`` (:func:`optimize_model`),
``model.int8.onnx`` (:func:`quantize_model`), ``model.fp16.onnx`` and
``image.jpg``. The FP16 copy keeps float32 inputs and outputs, so the SDK feeds
it the same tensors as the original. Quantization and FP16 conversion need
``onnx`` (the ``[quantize]`` extra, or the dev environment).

Calibration quality decides INT8 accuracy, not this script: point
``--calibration`` at a few dozen images from the deployment distribution. With
no ``--calibration``, the benchmark image alone is used — fine for timing,
meaningless for accuracy.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

from ort_vision_sdk import optimize_model, quantize_model

OUTPUT = Path(__file__).resolve().parent.parent / "bench" / "browser" / "models"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calibration_images(source: Path | None, fallback: Path) -> list[Path]:
    """List the calibration images.

    Args:
        source: A directory of images, or ``None``.
        fallback: The image used when no directory is given.

    Returns:
        The image paths, sorted.

    Raises:
        SystemExit: If ``source`` holds no image.
    """
    if source is None:
        return [fallback]
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images in {source}")
    return images


def to_fp16(model: Path, output: Path) -> None:
    """Convert a model to float16 weights and activations, float32 at the edges.

    Args:
        model: Source FP32 model.
        output: Destination.
    """
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    onnx.save(convert_float_to_float16(onnx.load(str(model)), keep_io_types=True), str(output))


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and write every variant.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="FP32 .onnx export")
    parser.add_argument("--image", type=Path, required=True, help="image the benchmark predicts on")
    parser.add_argument("--calibration", type=Path, default=None, help="directory of calibration images")
    parser.add_argument("--task", choices=["detect", "segment", "classify"], default=None)
    args = parser.parse_args(argv)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = OUTPUT / "model.onnx"
    shutil.copyfile(args.model, base)
    print(f"wrote {base}")

    Image.open(args.image).convert("RGB").save(OUTPUT / "image.jpg", quality=95)
    print(f"wrote {OUTPUT / 'image.jpg'}")

    optimize_model(base, OUTPUT / "model.opt.onnx")
    print(f"wrote {OUTPUT / 'model.opt.onnx'}")

    images = calibration_images(args.calibration, args.image)
    quantize_model(base, OUTPUT / "model.int8.onnx", images, task=args.task)
    print(f"wrote {OUTPUT / 'model.int8.onnx'} (calibrated on {len(images)} image(s))")

    to_fp16(base, OUTPUT / "model.fp16.onnx")
    print(f"wrote {OUTPUT / 'model.fp16.onnx'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
