"""Annotated-frame rendering: boxes, labels, zone polygon and the count overlay."""

from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import numpy as np

from .counter import FrameCounts
from .postprocess import Detections
from .zones import ZoneResult

__all__ = ["CLASS_COLORS", "draw_zone", "draw_detections", "draw_hud", "annotate_frame"]

# BGR, chosen to stay distinguishable on grey asphalt.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (0, 215, 255),      # amber
    "bicycle": (255, 191, 0),     # cyan-blue
    "car": (80, 220, 80),         # green
    "motorcycle": (255, 128, 255),  # pink
    "bus": (0, 128, 255),         # orange
    "truck": (200, 120, 60),      # steel blue
}
DEFAULT_COLOR = (200, 200, 200)
ZONE_COLOR = (255, 220, 120)
ZONE_COLOR_CONFLICT = (60, 60, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scaled(frame: np.ndarray) -> tuple[float, int]:
    """Font scale and line thickness that stay readable at any frame size."""
    base = max(frame.shape[0], frame.shape[1])
    scale = max(0.4, min(0.9, base / 1600.0))
    thickness = max(1, int(round(base / 800.0)))
    return scale, thickness


def draw_zone(frame: np.ndarray, zone, conflict: bool = False, alpha: float = 0.10) -> np.ndarray:
    """Fill + outline the ROI polygon. Turns red while a conflict is flagged."""
    if zone is None or not getattr(zone, "enabled", False):
        return frame
    pts = zone.polygon_int.reshape(-1, 1, 2)
    colour = ZONE_COLOR_CONFLICT if conflict else ZONE_COLOR
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)
    _, thickness = _scaled(frame)
    cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=thickness + 1)
    scale, _ = _scaled(frame)
    x, y = zone.polygon_int[0]
    cv2.putText(frame, "ZONE", (int(x) + 4, max(14, int(y) - 6)), FONT, scale * 0.7,
                colour, max(1, thickness), cv2.LINE_AA)
    return frame


def draw_detections(
    frame: np.ndarray,
    dets: Detections,
    label_name,
    in_zone: np.ndarray | None = None,
) -> np.ndarray:
    """Draw every detection. In-zone boxes get a thicker outline."""
    scale, thickness = _scaled(frame)
    boxes = np.asarray(dets.boxes, dtype=np.float32).reshape(-1, 4)
    for i in range(boxes.shape[0]):
        x0, y0, x1, y1 = (int(round(v)) for v in boxes[i])
        name = label_name(int(dets.labels[i]))
        colour = CLASS_COLORS.get(name, DEFAULT_COLOR)
        inside = bool(in_zone[i]) if in_zone is not None and in_zone.size > i else False
        box_thickness = thickness + 1 if inside else thickness
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, box_thickness)

        text = f"{name} {float(dets.scores[i]):.2f}"
        (tw, th), base = cv2.getTextSize(text, FONT, scale * 0.6, max(1, thickness))
        ty = y0 - 4 if y0 - th - base - 4 >= 0 else y1 + th + base + 2
        cv2.rectangle(
            frame,
            (x0, ty - th - base),
            (min(frame.shape[1] - 1, x0 + tw + 4), ty + base - 2),
            colour,
            -1,
        )
        cv2.putText(frame, text, (x0 + 2, ty - 2), FONT, scale * 0.6, (20, 20, 20),
                    max(1, thickness), cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    counts: FrameCounts,
    class_names: Sequence[str],
    zone_result: ZoneResult | None,
    frame_idx: int,
    timestamp_s: float,
    extra: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Translucent panel, top-left: frame id, per-class counts, zone counts, conflict."""
    scale, thickness = _scaled(frame)
    lines = [f"frame {frame_idx}  t={timestamp_s:.2f}s"]
    lines += [f"{name}: {counts.per_class[name]}" for name in class_names]
    lines.append(f"vehicle_total: {counts.vehicle_total}")
    if zone_result is not None:
        lines.append(f"in-zone persons: {zone_result.persons_in_zone}")
        lines.append(f"in-zone vehicles: {zone_result.vehicles_in_zone}")
    for key, value in (extra or {}).items():
        lines.append(f"{key}: {value}")

    line_h = int(round(22 * scale / 0.6))
    pad = int(round(8 * scale / 0.6))
    widths = [cv2.getTextSize(t, FONT, scale * 0.6, thickness)[0][0] for t in lines]
    panel_w = min(frame.shape[1] - 2 * pad, max(widths) + 2 * pad)
    panel_h = line_h * len(lines) + pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, dst=frame)

    y = pad + line_h - int(6 * scale / 0.6)
    for text in lines:
        cv2.putText(frame, text, (pad + 6, y), FONT, scale * 0.6, (240, 240, 240),
                    thickness, cv2.LINE_AA)
        y += line_h

    if zone_result is not None and zone_result.conflict:
        banner = "CONFLICT: pedestrian & vehicle in zone"
        (tw, th), base = cv2.getTextSize(banner, FONT, scale * 0.8, thickness + 1)
        x = max(pad, (frame.shape[1] - tw) // 2)
        y0 = frame.shape[0] - th - base - pad
        cv2.rectangle(frame, (x - 8, y0 - 6), (x + tw + 8, y0 + th + base), (40, 40, 220), -1)
        cv2.putText(frame, banner, (x, y0 + th), FONT, scale * 0.8, (255, 255, 255),
                    thickness + 1, cv2.LINE_AA)
    return frame


def annotate_frame(
    frame_bgr: np.ndarray,
    dets: Detections,
    counts: FrameCounts,
    class_names: Sequence[str],
    label_name,
    zone=None,
    zone_result: ZoneResult | None = None,
    frame_idx: int = 0,
    timestamp_s: float = 0.0,
    extra: Mapping[str, str] | None = None,
) -> np.ndarray:
    """One call to produce a fully annotated copy of a frame."""
    out = frame_bgr.copy()
    conflict = bool(zone_result.conflict) if zone_result is not None else False
    draw_zone(out, zone, conflict=conflict)
    draw_detections(out, dets, label_name,
                    in_zone=zone_result.mask if zone_result is not None else None)
    draw_hud(out, counts, class_names, zone_result, frame_idx, timestamp_s, extra)
    return out
