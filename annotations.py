"""Shared helpers for loading Label Studio annotation exports and mapping labels to windows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

AnnotationSegment = Tuple[float, float, Tuple[str, ...]]
AnnotationMap = Dict[str, List[AnnotationSegment]]
_DEFAULT_AUDIO_SUFFIX = r"(?:\.m4a)?$"


def load_annotations(annotation_file: Path) -> AnnotationMap:
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotation_file}")

    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    annotations: AnnotationMap = {}
    for entry in data:
        audio_name = entry.get("audio")
        if not isinstance(audio_name, str):
            continue
        audio_key = Path(audio_name).name
        segments = annotations.setdefault(audio_key, [])
        for label_entry in entry.get("label", []):
            try:
                start = float(label_entry.get("start", 0.0))
                end = float(label_entry.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            raw_labels = label_entry.get("labels", []) or []
            normalized_labels = tuple(
                label.strip().upper()
                for label in raw_labels
                if isinstance(label, str) and label.strip()
            )
            if end <= start or not normalized_labels:
                continue
            segments.append((start, end, normalized_labels))
    return annotations


def match_annotation_segments(annotation_map: AnnotationMap, sample_key: str, *, extension_pattern: str | None = None) -> List[AnnotationSegment]:
    """Return the first annotation segment list whose filename suffix matches the given key."""

    suffix_pattern = extension_pattern or _DEFAULT_AUDIO_SUFFIX
    pattern = re.compile(re.escape(sample_key) + suffix_pattern, re.IGNORECASE)
    for audio_name, segments in annotation_map.items():
        if pattern.search(audio_name):
            return segments
    return []


def labels_for_window(
    annotation_map: AnnotationMap,
    annotation_key: str,
    start_seconds: float,
    duration: float,
    *,
    extension_pattern: str | None = None,
) -> List[str]:
    segments = match_annotation_segments(annotation_map, annotation_key, extension_pattern=extension_pattern)
    if not segments:
        return []
    window_start = start_seconds
    window_end = start_seconds + duration

    found: List[str] = []
    seen = set()
    for seg_start, seg_end, labels_tuple in segments:
        if seg_start >= window_end or seg_end <= window_start:
            continue
        for label in labels_tuple:
            if label not in seen:
                seen.add(label)
                found.append(label)
    return found


def encode_labels(
    labels: Sequence[str],
    class_to_index: Mapping[str, int],
    num_classes: int,
) -> np.ndarray | None:
    if not labels:
        return None
    encoded = np.zeros(num_classes, dtype=np.float32)
    hit = False
    for label in labels:
        idx = class_to_index.get(label)
        if idx is None:
            continue
        encoded[idx] = 1.0
        hit = True
    return encoded if hit else None
