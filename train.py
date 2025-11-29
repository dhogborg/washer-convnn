"""Train a CNN to classify spectrogram features using annotation JSON files."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, cast

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from annotations import AnnotationMap, encode_labels, labels_for_window, load_annotations
from conv2d_model import CLASS_NAMES, CLASS_TO_INDEX, DEFAULT_IMAGE_SIZE, ConvClassifier, build_base_transform

DEFAULT_DATA_DIR = Path("./data_01")
DEFAULT_ANNOTATIONS_FILE = Path("./annotations/annotations.json")
SEGMENT_SECONDS = 10.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN on spectrogram PNG dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing spectrogram PNG files")
    parser.add_argument("--annotations-file", type=Path, default=DEFAULT_ANNOTATIONS_FILE, help="Path to the annotations JSON file")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of samples reserved for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and weight init")
    parser.add_argument("--img-size", type=int, nargs=2, default=DEFAULT_IMAGE_SIZE, metavar=("H", "W"), help="Resize height and width applied to all spectrograms")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--output", type=Path, default=Path("model.pt"), help="Path to save the best checkpoint")
    return parser.parse_args()


def parse_sample_metadata(image_path: Path) -> Tuple[str, float]:
    base = image_path.stem
    if "_" not in base:
        raise ValueError(f"Filename '{image_path.name}' missing annotation prefix")
    annotation_key, start_str = base.rsplit("_", 1)
    try:
        start_seconds = float(start_str)
    except ValueError as exc:
        raise ValueError(f"Invalid start seconds in '{image_path.name}'") from exc
    return annotation_key, start_seconds


@dataclass
class Sample:
    image_path: Path
    label: np.ndarray


def discover_samples(data_dir: Path, annotation_map: AnnotationMap) -> List[Sample]:
    samples: List[Sample] = []
    for png_path in sorted(data_dir.rglob("*.png")):
        try:
            annotation_key, start_seconds = parse_sample_metadata(png_path)
        except ValueError as exc:
            print(f"Skipping {png_path.name}: {exc}")
            continue

        labels = labels_for_window(annotation_map, annotation_key, start_seconds, SEGMENT_SECONDS)

        encoded = encode_labels(labels, CLASS_TO_INDEX, len(CLASS_NAMES))
        if encoded is None:
            continue
        samples.append(Sample(image_path=png_path, label=encoded))
    return samples


def train_val_split(samples: Sequence[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    if not samples:
        raise ValueError("No samples found in data directory")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - val_ratio)))
    train_samples = shuffled[:split_idx]
    val_samples = shuffled[split_idx:]
    if not val_samples:
        val_samples = train_samples[-1:]
        train_samples = train_samples[:-1]
    return train_samples, val_samples


class SpectrogramDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: transforms.Compose | None = None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("L")
        tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        tensor = cast(torch.Tensor, tensor)
        label_tensor = torch.from_numpy(sample.label)
        return tensor, label_tensor



def create_dataloaders(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    base_transforms = build_base_transform(img_size)

    train_dataset = SpectrogramDataset(train_samples, base_transforms)
    val_dataset = SpectrogramDataset(val_samples, base_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / max(total, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            correct += (preds == targets).float().mean(dim=1).sum().item()

    avg_loss = running_loss / max(total, 1)
    avg_accuracy = correct / max(total, 1)
    return avg_loss, avg_accuracy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    annotations_file = args.annotations_file.expanduser().resolve()
    annotation_map = load_annotations(annotations_file)
    samples = discover_samples(data_dir, annotation_map)
    train_samples, val_samples = train_val_split(samples, args.val_split, args.seed)

    train_loader, val_loader = create_dataloaders(
        train_samples,
        val_samples,
        tuple(args.img_size),
        args.batch_size,
        args.num_workers,
    )

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print("using device", device)
    model = ConvClassifier(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "classes": CLASS_NAMES}, args.output)
            print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()

