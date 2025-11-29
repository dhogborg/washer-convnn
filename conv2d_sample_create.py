"""Generate fixed-length spectrograms from an input m4a file."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
from PIL import Image
from scipy import signal

SAMPLE_RATE = 44_100
BLOCK_SECONDS = 10
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_SECONDS


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Create 10-second spectrograms from an .m4a file")
	parser.add_argument("--input", type=Path, help="Path to the source .m4a audio file")
	parser.add_argument("--output", type=Path, help="Directory where spectrogram PNGs will be written")
	return parser.parse_args()


def generate_spectrogram(samples: np.ndarray, samplerate: int, out_path: Path) -> None:
	"""Mirror the spectrogram style used in record.py."""
	_, _, Sxx = signal.spectrogram(samples, fs=samplerate, nperseg=1024, noverlap=512)
	sxx_db = 10 * np.log10(Sxx + 1e-12)

	min_val = np.min(sxx_db)
	max_val = np.max(sxx_db)
	if np.isclose(max_val, min_val):
		normalized = np.zeros_like(sxx_db)
	else:
		normalized = (sxx_db - min_val) / (max_val - min_val)

	image_array = np.flipud((1.0 - normalized) * 255.0).astype(np.uint8)
	Image.fromarray(image_array, mode="L").save(out_path)


def main() -> None:
	args = parse_args()
	input_path = args.input.expanduser().resolve()
	output_dir = args.output.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	audio, sr = librosa.load(input_path.as_posix(), sr=SAMPLE_RATE, mono=True)
	if sr != SAMPLE_RATE:
		raise RuntimeError(f"Unexpected resample mismatch: expected {SAMPLE_RATE}, got {sr}")

	total_samples = len(audio)
	if total_samples < BLOCK_SAMPLES:
		raise ValueError("Audio shorter than required 10-second block")

	base_name = input_path.stem
	block_count = total_samples // BLOCK_SAMPLES

	for block_idx in range(block_count):
		start = block_idx * BLOCK_SAMPLES
		end = start + BLOCK_SAMPLES
		chunk = audio[start:end]
		timestamp = block_idx * BLOCK_SECONDS
		out_path = output_dir / f"{base_name}_{timestamp}.png"
		generate_spectrogram(chunk, SAMPLE_RATE, out_path)
		print(f"Wrote {out_path}")

	print(f"Generated {block_count} spectrograms in {output_dir}")


if __name__ == "__main__":
	main()
