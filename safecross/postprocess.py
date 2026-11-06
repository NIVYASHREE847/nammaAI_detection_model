"""NumPy re-implementation of torchvision's SSD detection post-processing.

The exported .pte contains only the backbone + SSD head, so it emits raw
``cls_logits`` (N, num_anchors, num_classes) and ``bbox_regression``
(N, num_anchors, 4). Everything after that happens here, in NumPy, exactly as
``torchvision.models.detection.ssd.SSD.postprocess_detections`` does it:

1. softmax over the class dimension,
2. per class ``label >= 1``: threshold on score, keep the top
   ``topk_candidates``, decode the kept boxes against their anchors,
3. clip to the 320x320 network input,
4. class-aware NMS (torchvision's ``batched_nms`` coordinate-offset trick),
5. keep the top ``detections_per_img`` by score,
6. rescale boxes from 320x320 back to the original frame size
   (``GeneralizedRCNNTransform.postprocess`` -> ``resize_boxes``).

Doing this in NumPy keeps the exported graph static and free of NMS, which is
what makes the program portable to the XNNPACK CPU backend on a Pi.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np

__all__ = [
    "Detections",
    "softmax",
    "decode_boxes",
    "clip_boxes",
    "nms",
    "batched_nms",
    "resize_boxes",
    "postprocess_detections",
    "merge_detections",
    "suppress_cross_class_duplicates",
]

# torchvision det_utils.BoxCoder default: math.log(1000.0 / 16)
BBOX_XFORM_CLIP = math.log(1000.0 / 16)


class Detections(NamedTuple):
    """Detections for a single frame, in original-frame pixel coordinates."""

    boxes: np.ndarray   # (M, 4) float32, xyxy
    scores: np.ndarray  # (M,) float32
    labels: np.ndarray  # (M,) int64, torchvision/COCO label indices

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @staticmethod
    def empty() -> "Detections":
        return Detections(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = np.asarray(logits, dtype=np.float32) - np.max(logits, axis=axis, keepdims=True)
    np.exp(shifted, out=shifted)
    return shifted / np.sum(shifted, axis=axis, keepdims=True)


def decode_boxes(
    rel_codes: np.ndarray,
    anchors: np.ndarray,
    weights: Sequence[float] = (10.0, 10.0, 5.0, 5.0),
) -> np.ndarray:
    """Decode SSD box regressions against anchors.

    ``rel_codes`` is (M, 4) ``(dx, dy, dw, dh)``; ``anchors`` is (M, 4) xyxy in
    network-input pixels. Matches ``det_utils.BoxCoder.decode_single``.
    """
    rel_codes = np.asarray(rel_codes, dtype=np.float32).reshape(-1, 4)
    anchors = np.asarray(anchors, dtype=np.float32).reshape(-1, 4)
    if rel_codes.shape[0] != anchors.shape[0]:
        raise ValueError(
            f"regression/anchor count mismatch: {rel_codes.shape[0]} vs {anchors.shape[0]}"
        )

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = rel_codes[:, 0] / wx
    dy = rel_codes[:, 1] / wy
    dw = np.minimum(rel_codes[:, 2] / ww, BBOX_XFORM_CLIP)
    dh = np.minimum(rel_codes[:, 3] / wh, BBOX_XFORM_CLIP)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    half_w = 0.5 * pred_w
    half_h = 0.5 * pred_h
    out = np.empty((rel_codes.shape[0], 4), dtype=np.float32)
    out[:, 0] = pred_ctr_x - half_w
    out[:, 1] = pred_ctr_y - half_h
    out[:, 2] = pred_ctr_x + half_w
    out[:, 3] = pred_ctr_y + half_h
    return out


def clip_boxes(boxes: np.ndarray, height: float, width: float) -> np.ndarray:
    """Clip xyxy boxes into [0, width] x [0, height] (``clip_boxes_to_image``)."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    np.clip(boxes[:, 0], 0.0, width, out=boxes[:, 0])
    np.clip(boxes[:, 2], 0.0, width, out=boxes[:, 2])
    np.clip(boxes[:, 1], 0.0, height, out=boxes[:, 1])
    np.clip(boxes[:, 3], 0.0, height, out=boxes[:, 3])
    return boxes


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns kept indices, score-descending.

    Score ties break toward the lower index, matching torchvision's stable sort.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    order = np.argsort(-scores, kind="stable")

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(xx1 - xx0, 0.0) * np.maximum(yy1 - yy0, 0.0)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, idxs: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Class-aware NMS: boxes with different ``idxs`` never suppress each other.

    Same trick as ``torchvision.ops.batched_nms``: offset each group's
    coordinates by ``group_id * (max_coordinate + 1)`` so groups cannot overlap,
    then run a single NMS.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    idxs = np.asarray(idxs).reshape(-1)
    offsets = idxs.astype(np.float32) * (float(boxes.max()) + 1.0)
    return nms(boxes + offsets[:, None], scores, iou_threshold)


def resize_boxes(
    boxes: np.ndarray,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> np.ndarray:
    """Rescale xyxy boxes between two image sizes, both given as (height, width).

    Mirrors ``torchvision.models.detection.transform.resize_boxes``.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return boxes.copy()
    ratio_h = to_size[0] / from_size[0]
    ratio_w = to_size[1] / from_size[1]
    out = boxes.copy()
    out[:, 0] *= ratio_w
    out[:, 2] *= ratio_w
    out[:, 1] *= ratio_h
    out[:, 3] *= ratio_h
    return out


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, score-descending (ties: lower index)."""
    k = min(int(k), scores.shape[0])
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    if k == scores.shape[0]:
        return np.argsort(-scores, kind="stable")
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


def postprocess_detections(
    cls_logits: np.ndarray,
    bbox_regression: np.ndarray,
    anchors: np.ndarray,
    *,
    input_size: tuple[int, int] = (320, 320),
    original_size: tuple[int, int] | None = None,
    score_threshold: float = 0.5,
    nms_iou: float = 0.45,
    topk_candidates: int = 300,
    detections_per_img: int = 300,
    keep_labels: Sequence[int] | None = None,
    box_weights: Sequence[float] = (10.0, 10.0, 5.0, 5.0),
) -> Detections:
    """Turn raw head outputs for ONE image into final detections.

    ``input_size`` and ``original_size`` are (height, width). When
    ``original_size`` is given, boxes come back in original-frame pixels.
    ``keep_labels`` restricts which label indices are considered at all, which
    both matches what we report and saves time (6 classes instead of 90).
    """
    cls_logits = np.asarray(cls_logits, dtype=np.float32)
    bbox_regression = np.asarray(bbox_regression, dtype=np.float32)
    if cls_logits.ndim == 3:
        if cls_logits.shape[0] != 1:
            raise ValueError("postprocess_detections handles one image at a time")
        cls_logits = cls_logits[0]
        bbox_regression = bbox_regression[0]
    anchors = np.asarray(anchors, dtype=np.float32).reshape(-1, 4)

    num_classes = cls_logits.shape[-1]
    scores_all = softmax(cls_logits, axis=-1)
    labels_iter = (
        list(range(1, num_classes))
        if keep_labels is None
        else sorted({int(x) for x in keep_labels})
    )

    box_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    for label in labels_iter:
        if label < 1 or label >= num_classes:
            raise ValueError(f"label {label} outside the model's 1..{num_classes - 1} range")
        score = scores_all[:, label]
        keep_idxs = np.nonzero(score > score_threshold)[0]
        if keep_idxs.size == 0:
            continue
        score_kept = score[keep_idxs]
        top = _topk(score_kept, topk_candidates)
        sel = keep_idxs[top]
        # Decode only the surviving anchors -- cheaper than decoding all 3000+.
        boxes = decode_boxes(bbox_regression[sel], anchors[sel], box_weights)
        boxes = clip_boxes(boxes, input_size[0], input_size[1])
        box_chunks.append(boxes)
        score_chunks.append(score_kept[top])
        label_chunks.append(np.full(sel.shape[0], label, dtype=np.int64))

    if not box_chunks:
        return Detections.empty()

    boxes = np.concatenate(box_chunks, axis=0)
    scores = np.concatenate(score_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)

    keep = batched_nms(boxes, scores, labels, nms_iou)[:detections_per_img]
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if original_size is not None:
        boxes = resize_boxes(boxes, input_size, original_size)
    return Detections(boxes.astype(np.float32), scores.astype(np.float32), labels)


def suppress_cross_class_duplicates(
    dets: Detections,
    group_labels: Sequence[int],
    iou_threshold: float = 0.6,
) -> Detections:
    """Drop lower-scoring duplicates of one object that got two different labels.

    torchvision's SSD post-processing runs NMS per class, so a single car can
    survive as both ``car 0.52`` and ``truck 0.70``. That is harmless for a
    detection benchmark but it double-counts for ``vehicle_total``, and tiled
    inference makes it more common because neighbouring tiles see the same
    object from different crops.

    This runs one extra class-AGNOSTIC NMS restricted to ``group_labels`` (the
    vehicle classes), on the argument that one physical vehicle cannot be two
    vehicle types at once. Detections outside the group -- notably ``person``
    -- are left untouched, so a pedestrian standing against a car is never
    suppressed.
    """
    labels = np.asarray(dets.labels).reshape(-1)
    if labels.size < 2 or not group_labels:
        return dets
    in_group = np.isin(labels, list(group_labels))
    group_idx = np.nonzero(in_group)[0]
    if group_idx.size < 2:
        return dets
    kept_group = group_idx[nms(dets.boxes[group_idx], dets.scores[group_idx], iou_threshold)]
    keep = np.concatenate([np.nonzero(~in_group)[0], kept_group])
    keep = keep[np.argsort(-dets.scores[keep], kind="stable")]
    return Detections(dets.boxes[keep], dets.scores[keep], labels[keep])


def merge_detections(
    parts: Sequence[Detections],
    nms_iou: float = 0.45,
    detections_per_img: int = 300,
) -> Detections:
    """Concatenate detections from several tiles and de-duplicate with class-aware NMS."""
    kept = [p for p in parts if len(p) > 0]
    if not kept:
        return Detections.empty()
    boxes = np.concatenate([p.boxes for p in kept], axis=0)
    scores = np.concatenate([p.scores for p in kept], axis=0)
    labels = np.concatenate([p.labels for p in kept], axis=0)
    keep = batched_nms(boxes, scores, labels, nms_iou)[:detections_per_img]
    return Detections(boxes[keep], scores[keep], labels[keep])
