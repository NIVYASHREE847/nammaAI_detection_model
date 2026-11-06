"""Polygon region of interest, in-zone counting and the conflict rule.

Why a zone at all: a raw frame count says "6 people, 3 cars somewhere in
view". For pedestrian-safety evidence what matters is "people and vehicles
occupying the same piece of road at the same instant". The polygon marks that
piece of road (a crossing, a bus-stop frontage, a school gate), and a detection
counts as inside it when its ground-contact point -- the bottom centre of the
box -- falls inside the polygon. Using the box centre instead would place a
tall pedestrian standing at the kerb inside a road zone, so bottom-centre is
the default.

The polygon is stored in config.yaml in normalized [0, 1] coordinates, so the
same configuration is valid for any capture resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .postprocess import Detections

__all__ = ["Zone", "ZoneResult", "ConflictRule", "anchor_points", "points_in_polygon"]

ANCHOR_MODES = ("bottom_center", "center", "top_center")


def anchor_points(boxes: np.ndarray, mode: str = "bottom_center") -> np.ndarray:
    """Representative point per xyxy box, shape (M, 2) as (x, y)."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    if mode == "bottom_center":
        y = boxes[:, 3]
    elif mode == "center":
        y = 0.5 * (boxes[:, 1] + boxes[:, 3])
    elif mode == "top_center":
        y = boxes[:, 1]
    else:
        raise ValueError(f"unknown anchor_point {mode!r}; expected one of {ANCHOR_MODES}")
    return np.stack([x, y], axis=1)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd (ray casting) test. Returns a boolean mask.

    Points exactly on an edge are not guaranteed either way, which is
    acceptable here: a detection box edge is far noisier than one pixel.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if polygon.shape[0] < 3:
        raise ValueError("a polygon needs at least 3 vertices")
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    n = polygon.shape[0]
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[i - 1]  # i-1 wraps to the last vertex when i == 0
        dy = yj - yi
        # A horizontal edge can never be crossed by a horizontal ray: the
        # straddle test below is already False when dy == 0, so guard the
        # division only to keep NumPy from warning.
        straddles = (yi > y) != (yj > y)
        if not np.any(straddles):
            continue
        safe_dy = dy if dy != 0.0 else 1.0
        x_at_edge = (xj - xi) * (y - yi) / safe_dy + xi
        inside ^= straddles & (x < x_at_edge)
    return inside


@dataclass(frozen=True)
class ZoneResult:
    """Outcome of applying a zone + conflict rule to one frame."""

    mask: np.ndarray            # (M,) bool, which detections are inside the zone
    persons_in_zone: int
    vehicles_in_zone: int
    conflict: bool              # the reported flag (after the min-duration rule)
    condition: bool = False     # the raw count condition, before min-duration
    streak: int = 0             # how many consecutive frames the condition has held


class ConflictRule:
    """Count thresholds plus a minimum-duration rule, evaluated frame by frame.

    The raw condition is ``persons_in_zone >= P AND vehicles_in_zone >= V``. On
    a busy crossing that condition is true almost always, and a flag that is
    always on carries no information. The minimum-duration rule fixes the other
    half of the problem: a single frame where a detector flickers a person into
    the zone is noise, while the same condition holding for half a second is a
    real co-presence.

    The flag is raised once the condition has held for ``min_consecutive_frames``
    **processed** frames in a row, and stays raised for as long as it keeps
    holding. The first ``min_consecutive_frames - 1`` frames of an episode are
    therefore NOT flagged: the rule is causal, so the annotated frame, the CSV
    row and a live stream all agree without a second pass. Episode starts are
    recorded separately (``episodes``), so the report can still say when each
    episode actually began.

    ``min_consecutive_frames`` counts processed frames, so it interacts with
    ``--stride``: at stride 5 and 30 fps, 5 frames is 0.83 s, not 0.17 s.
    """

    def __init__(
        self,
        min_persons: int = 1,
        min_vehicles: int = 1,
        min_consecutive_frames: int = 1,
    ) -> None:
        self.min_persons = int(min_persons)
        self.min_vehicles = int(min_vehicles)
        self.min_consecutive_frames = max(1, int(min_consecutive_frames))
        self.streak = 0
        self.episodes: list[dict] = []
        self._open: dict | None = None

    def update(self, frame_idx: int, timestamp_s: float,
               persons: int, vehicles: int) -> tuple[bool, bool]:
        """Feed one frame. Returns ``(flag, raw_condition)``."""
        condition = persons >= self.min_persons and vehicles >= self.min_vehicles
        if condition:
            self.streak += 1
            if self._open is None:
                self._open = {
                    "start_frame": int(frame_idx),
                    "start_s": float(timestamp_s),
                    "end_frame": int(frame_idx),
                    "end_s": float(timestamp_s),
                    "frames": 0,
                    "max_persons": 0,
                    "max_vehicles": 0,
                }
            self._open["end_frame"] = int(frame_idx)
            self._open["end_s"] = float(timestamp_s)
            self._open["frames"] += 1
            self._open["max_persons"] = max(self._open["max_persons"], int(persons))
            self._open["max_vehicles"] = max(self._open["max_vehicles"], int(vehicles))
        else:
            self._close()
            self.streak = 0
        return self.streak >= self.min_consecutive_frames, condition

    def _close(self) -> None:
        if self._open is not None:
            if self._open["frames"] >= self.min_consecutive_frames:
                self.episodes.append(self._open)
            self._open = None

    def finish(self) -> list[dict]:
        """Close any open episode and return the qualifying episodes."""
        self._close()
        return self.episodes

    def describe(self) -> str:
        duration = (
            "" if self.min_consecutive_frames <= 1
            else f", sustained for >= {self.min_consecutive_frames} consecutive processed frames"
        )
        return (f"persons_in_zone >= {self.min_persons} AND "
                f"vehicles_in_zone >= {self.min_vehicles}{duration}")

    @classmethod
    def from_config(cls, cfg: Mapping) -> "ConflictRule":
        ccfg = cfg.get("conflict", {}) or {}
        return cls(
            min_persons=int(ccfg.get("min_persons", 1)),
            min_vehicles=int(ccfg.get("min_vehicles", 1)),
            min_consecutive_frames=int(ccfg.get("min_consecutive_frames", 1)),
        )


class Zone:
    """A pixel-space polygon ROI plus the conflict rule evaluated inside it."""

    def __init__(
        self,
        polygon: np.ndarray,
        anchor_mode: str = "bottom_center",
        min_persons: int = 1,
        min_vehicles: int = 1,
        enabled: bool = True,
    ) -> None:
        if anchor_mode not in ANCHOR_MODES:
            raise ValueError(f"unknown anchor_point {anchor_mode!r}; expected {ANCHOR_MODES}")
        self.polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if enabled and self.polygon.shape[0] < 3:
            raise ValueError("an enabled zone needs at least 3 polygon vertices")
        self.anchor_mode = anchor_mode
        self.min_persons = int(min_persons)
        self.min_vehicles = int(min_vehicles)
        self.enabled = bool(enabled)

    @classmethod
    def from_config(cls, cfg: Mapping, width: int, height: int) -> "Zone":
        """Build from config.yaml, scaling normalized vertices to pixels."""
        zcfg = cfg.get("zone", {}) or {}
        ccfg = cfg.get("conflict", {}) or {}
        norm = np.asarray(zcfg.get("polygon_norm") or [], dtype=np.float32).reshape(-1, 2)
        enabled = bool(zcfg.get("enabled", True)) and norm.shape[0] >= 3
        polygon = norm * np.asarray([[width, height]], dtype=np.float32)
        return cls(
            polygon=polygon,
            anchor_mode=str(zcfg.get("anchor_point", "bottom_center")),
            min_persons=int(ccfg.get("min_persons", 1)),
            min_vehicles=int(ccfg.get("min_vehicles", 1)),
            enabled=enabled,
        )

    def contains_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Boolean mask: which boxes have their anchor point inside the polygon."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if not self.enabled:
            # Zone disabled: the whole frame is the zone.
            return np.ones(boxes.shape[0], dtype=bool)
        return points_in_polygon(anchor_points(boxes, self.anchor_mode), self.polygon)

    def evaluate(self, dets: Detections, counter, rule: "ConflictRule | None" = None,
                 frame_idx: int = 0, timestamp_s: float = 0.0) -> ZoneResult:
        """Count persons/vehicles inside the zone and apply the conflict rule.

        Pass ``rule`` to use the configured thresholds plus the minimum-duration
        rule. Without it the zone falls back to its own ``min_persons`` /
        ``min_vehicles`` with no duration requirement, which is what the unit
        tests exercise.
        """
        mask = self.contains_boxes(dets.boxes)
        labels = np.asarray(dets.labels).reshape(-1)[mask]
        inside = Detections(dets.boxes[mask], dets.scores[mask], labels)
        persons, vehicles = counter.person_vehicle_split(inside)
        if rule is None:
            condition = persons >= self.min_persons and vehicles >= self.min_vehicles
            return ZoneResult(mask, persons, vehicles, bool(condition),
                              bool(condition), 1 if condition else 0)
        flag, condition = rule.update(frame_idx, timestamp_s, persons, vehicles)
        return ZoneResult(mask, persons, vehicles, bool(flag), bool(condition), rule.streak)

    @property
    def polygon_int(self) -> np.ndarray:
        """Vertices rounded to int32, ready for ``cv2.polylines``."""
        return np.round(self.polygon).astype(np.int32)

    def describe(self, rule: "ConflictRule | None" = None) -> str:
        if not self.enabled:
            return "disabled (whole frame counted as the zone)"
        pts = ", ".join(f"({int(x)},{int(y)})" for x, y in self.polygon)
        condition = (rule.describe() if rule is not None
                     else f"persons_in_zone >= {self.min_persons} AND "
                          f"vehicles_in_zone >= {self.min_vehicles}")
        return (
            f"{self.polygon.shape[0]}-vertex polygon [{pts}], "
            f"anchor={self.anchor_mode}, conflict when {condition}"
        )
