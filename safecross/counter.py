"""Turn detections into the per-class counts the report is built from.

This is a per-frame instantaneous count, not a unique-object count: the same
pedestrian walking across the frame is counted again in every frame. That is
the right measure for the counting task in the problem statement ("how many
objects are in view now"), and it is what the CSV and the overlay show. It is
NOT a footfall total -- summing the CSV column over frames would multiply every
object by how long it stayed on screen. Unique-object totals would need
tracking, which is out of scope for this PoC and is called out in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import COCO_INSTANCE_CATEGORY_NAMES, coco_name_to_id
from .postprocess import Detections

__all__ = ["ClassCounter", "FrameCounts"]


@dataclass(frozen=True)
class FrameCounts:
    """Counts for one frame."""

    per_class: Mapping[str, int]
    vehicle_total: int
    total: int

    def as_row(self, class_names: Sequence[str]) -> list[int]:
        return [int(self.per_class[name]) for name in class_names] + [int(self.vehicle_total)]


class ClassCounter:
    """Counts detections per configured class and rolls up ``vehicle_total``."""

    def __init__(
        self,
        class_names: Sequence[str],
        vehicle_classes: Sequence[str],
        categories: Sequence[str] | None = None,
    ) -> None:
        categories = list(categories or COCO_INSTANCE_CATEGORY_NAMES)
        self.class_names = list(class_names)
        unknown = [c for c in self.class_names if c not in categories]
        if unknown:
            raise ValueError(f"classes not in the model's label set: {unknown}")
        bad_vehicles = [c for c in vehicle_classes if c not in self.class_names]
        if bad_vehicles:
            raise ValueError(
                f"vehicle_classes must be a subset of classes; extra: {bad_vehicles}"
            )
        self.vehicle_classes = list(vehicle_classes)
        self.id_by_name = {name: categories.index(name) for name in self.class_names}
        self.name_by_id = {idx: name for name, idx in self.id_by_name.items()}
        self.keep_labels = sorted(self.id_by_name.values())
        self.vehicle_ids = {self.id_by_name[name] for name in self.vehicle_classes}

    # -- counting -----------------------------------------------------------
    def count(self, dets: Detections) -> FrameCounts:
        labels = np.asarray(dets.labels).reshape(-1)
        per_class = {}
        for name, label_id in self.id_by_name.items():
            per_class[name] = int(np.count_nonzero(labels == label_id))
        vehicle_total = int(sum(per_class[name] for name in self.vehicle_classes))
        return FrameCounts(per_class, vehicle_total, int(labels.size))

    def count_subset(self, dets: Detections, mask: np.ndarray) -> FrameCounts:
        """Count only the detections selected by a boolean mask (e.g. in-zone)."""
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        return self.count(
            Detections(dets.boxes[mask], dets.scores[mask], np.asarray(dets.labels)[mask])
        )

    def person_vehicle_split(self, dets: Detections) -> tuple[int, int]:
        """``(persons, vehicles)`` -- the two numbers the conflict rule needs."""
        labels = np.asarray(dets.labels).reshape(-1)
        person_id = coco_name_to_id("person")
        persons = int(np.count_nonzero(labels == person_id))
        vehicles = int(np.count_nonzero(np.isin(labels, list(self.vehicle_ids))))
        return persons, vehicles

    # -- presentation -------------------------------------------------------
    def label_name(self, label_id: int) -> str:
        return self.name_by_id.get(int(label_id), f"class_{int(label_id)}")

    @property
    def csv_columns(self) -> list[str]:
        return [*self.class_names, "vehicle_total"]

    @classmethod
    def from_config(cls, cfg: Mapping, categories: Sequence[str] | None = None) -> "ClassCounter":
        det = cfg["detection"]
        return cls(det["classes"], det["vehicle_classes"], categories)
