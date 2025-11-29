"""Evaluate a trained spectrogram classifier on a single PNG input."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch

from conv2d_model import CLASS_NAMES, DEFAULT_IMAGE_SIZE, build_base_transform, load_weights


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run inference on a single spectrogram PNG")
	parser.add_argument("--model", type=Path, required=True, help="Path to a trained model checkpoint (.pt)")
	parser.add_argument("--input", type=Path, required=True, help="Path to the spectrogram PNG to classify")
	parser.add_argument("--img-size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE, metavar=("H", "W"), help="Resize height and width applied before inference")
	parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Explicit device override (defaults to auto-detect)")
	return parser.parse_args()


def resolve_device(preferred: str | None) -> torch.device:
	if preferred:
		return torch.device(preferred)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch, "mps") and torch.mps.is_available():  # type: ignore[attr-defined]
		return torch.device("mps")
	return torch.device("cpu")


def load_image_tensor(image_path: Path, transform, device: torch.device) -> torch.Tensor:
	image = Image.open(image_path).convert("L")
	tensor = transform(image).unsqueeze(0)
	return tensor.to(device)


def format_scores(classes: Sequence[str], scores: np.ndarray) -> str:
	rows = [f"{label:>12s}: {score:.4f}" for label, score in zip(classes, scores)]
	return "\n".join(rows)


def main() -> None:
	args = parse_args()
	device = resolve_device(args.device)

	model_path = args.model.expanduser().resolve()
	spectrogram_path = args.input.expanduser().resolve()

	transform = build_base_transform(tuple(args.img_size))
	model, classes = load_weights(model_path, device)
	model.eval()

	tensor = load_image_tensor(spectrogram_path, transform, device)
	with torch.no_grad():
		logits = model(tensor)
		probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

	if len(classes) != len(probs):
		raise ValueError("Mismatch between model classes and output scores")

	print(format_scores(classes, probs))


if __name__ == "__main__":
	main()
