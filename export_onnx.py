"""Export trained checkpoints to ONNX for lightweight deployment (e.g., Raspberry Pi)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Tuple

import torch

from conv2d_model import CLASS_NAMES, DEFAULT_IMAGE_SIZE, load_weights as load_conv2d_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export audio classifiers to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the PyTorch checkpoint (.pt)")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination ONNX file (e.g., conv2d_model.onnx)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=DEFAULT_IMAGE_SIZE,
        metavar=("H", "W"),
        help="Input size used for conv2d exports",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    return parser.parse_args()


def export_conv2d(checkpoint: Path, output: Path, img_size: Tuple[int, int], opset: int) -> None:
    device = torch.device("cpu")
    model, classes = load_conv2d_weights(checkpoint, device)
    model.eval()

    dummy = torch.randn(1, 1, img_size[0], img_size[1], device=device)
    torch.onnx.export(
        model,
        (dummy,),
        output.as_posix(),
        input_names=["spectrogram"],
        output_names=["logits"],
        dynamic_axes={"spectrogram": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    write_metadata(output, classes)


def write_metadata(output_path: Path, classes: Sequence[str]) -> None:
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    content = {
        "classes": list(classes) if classes else CLASS_NAMES,
        "source": output_path.name,
    }
    meta_path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    export_conv2d(checkpoint, output, tuple(args.img_size), args.opset)
    
    print(f"Exported ONNX model to {output}")


if __name__ == "__main__":
    main()
