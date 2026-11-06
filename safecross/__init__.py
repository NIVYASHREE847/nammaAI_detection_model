"""SafeCross: edge-ready object detection and counting for pedestrian-safety evidence.

The package is deliberately split so that nothing but ``runtime`` needs a deep
learning framework at inference time:

* ``preprocess``  -- uint8 frame -> normalized NCHW float32 tensor (NumPy only)
* ``runtime``     -- runs one tensor through a model (ExecuTorch .pte or eager torch)
* ``postprocess`` -- anchor decoding, score threshold and NMS (NumPy only)
* ``counter``     -- detections -> per-class counts
* ``zones``       -- polygon region of interest and in-zone counting
* ``visualize``   -- annotated frames
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "load_config",
    "COCO_INSTANCE_CATEGORY_NAMES",
    "coco_name_to_id",
    "ROOT",
]

ROOT = Path(__file__).resolve().parent.parent

# torchvision's COCO detection models predict label indices into this 91-entry
# table (index 0 is background, and the table keeps COCO's numbering gaps as
# "N/A"). Keeping the full table means a label id can always be printed.
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def coco_name_to_id(name: str) -> int:
    """Return the label index torchvision predicts for a COCO class name."""
    try:
        return COCO_INSTANCE_CATEGORY_NAMES.index(name)
    except ValueError as exc:  # pragma: no cover - configuration error
        raise KeyError(f"unknown COCO class name: {name!r}") from exc


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load config.yaml. Paths inside it are resolved relative to the repo root."""
    import yaml  # imported lazily so NumPy-only consumers need no PyYAML

    path = Path(path)
    if not path.is_absolute():
        candidate = Path.cwd() / path
        path = candidate if candidate.exists() else ROOT / path
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    return cfg


def load_meta(path: str | Path) -> dict[str, Any]:
    """Load the JSON metadata written beside the .pte files at export time."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve(path: str | Path) -> Path:
    """Resolve a config path against the repo root when it is relative."""
    path = Path(path)
    if path.is_absolute():
        return path
    local = Path.cwd() / path
    return local if local.exists() else ROOT / path
