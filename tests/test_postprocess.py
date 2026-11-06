"""Tests for the NumPy post-processing: decoding, clipping, NMS, merging."""

from __future__ import annotations

import math

import numpy as np
import pytest

from safecross.postprocess import (
    BBOX_XFORM_CLIP,
    Detections,
    batched_nms,
    clip_boxes,
    decode_boxes,
    merge_detections,
    nms,
    postprocess_detections,
    resize_boxes,
    softmax,
    suppress_cross_class_duplicates,
)


def test_softmax_sums_to_one_and_is_shift_invariant():
    logits = np.array([[1.0, 2.0, 3.0], [-50.0, 0.0, 50.0]], dtype=np.float32)
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert np.allclose(probs, softmax(logits + 1000.0), atol=1e-6)
    assert np.all(np.isfinite(probs))


def test_decode_zero_regression_returns_the_anchor():
    anchors = np.array([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    decoded = decode_boxes(np.zeros((2, 4), dtype=np.float32), anchors)
    assert np.allclose(decoded, anchors, atol=1e-5)


def test_decode_matches_hand_computed_shift_and_scale():
    # One anchor: centre (30, 50), size 40 x 60.
    anchors = np.array([[10.0, 20.0, 50.0, 80.0]], dtype=np.float32)
    weights = (10.0, 10.0, 5.0, 5.0)
    rel = np.array([[1.0, -2.0, 5.0, 0.0]], dtype=np.float32)
    # dx = 0.1 -> ctr_x += 0.1 * 40 = 34; dy = -0.2 -> ctr_y -= 0.2 * 60 = 38
    # dw = 1.0 -> w = e * 40; dh = 0 -> h = 60
    out = decode_boxes(rel, anchors, weights)[0]
    width = math.e * 40.0
    expected = [34.0 - width / 2, 38.0 - 30.0, 34.0 + width / 2, 38.0 + 30.0]
    assert np.allclose(out, expected, atol=1e-3)


def test_decode_clamps_extreme_growth():
    anchors = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    huge = np.array([[0.0, 0.0, 1e4, 1e4]], dtype=np.float32)
    out = decode_boxes(huge, anchors)
    max_side = math.exp(BBOX_XFORM_CLIP) * 10.0
    assert np.all(np.isfinite(out))
    assert (out[0, 2] - out[0, 0]) <= max_side + 1e-3


def test_decode_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        decode_boxes(np.zeros((3, 4), dtype=np.float32), np.zeros((2, 4), dtype=np.float32))


def test_clip_boxes_bounds_coordinates():
    boxes = np.array([[-5.0, -5.0, 400.0, 400.0]], dtype=np.float32)
    out = clip_boxes(boxes, 320, 320)
    assert out.tolist() == [[0.0, 0.0, 320.0, 320.0]]


def test_nms_keeps_best_and_drops_overlap():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2]


def test_nms_keeps_both_when_iou_below_threshold():
    boxes = np.array([[0, 0, 10, 10], [6, 0, 16, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    # IoU = 4/16 = 0.25
    assert nms(boxes, scores, 0.5).tolist() == [0, 1]
    assert nms(boxes, scores, 0.1).tolist() == [0]


def test_nms_on_empty_input():
    keep = nms(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.5)
    assert keep.shape == (0,)


def test_batched_nms_does_not_suppress_across_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    same_class = batched_nms(boxes, scores, np.array([1, 1]), 0.5)
    other_class = batched_nms(boxes, scores, np.array([1, 3]), 0.5)
    assert same_class.tolist() == [0]
    assert sorted(other_class.tolist()) == [0, 1]


def test_resize_boxes_scales_each_axis_independently():
    boxes = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    out = resize_boxes(boxes, (320, 320), (640, 160))
    assert np.allclose(out, [[5.0, 40.0, 15.0, 80.0]])


def test_postprocess_finds_a_planted_detection():
    """One anchor is given a dominant class score; it must come back decoded."""
    num_anchors, num_classes = 8, 91
    anchors = np.tile(np.array([[100.0, 100.0, 140.0, 180.0]], dtype=np.float32),
                      (num_anchors, 1))
    logits = np.full((num_anchors, num_classes), -10.0, dtype=np.float32)
    logits[:, 0] = 10.0            # everything is background ...
    logits[3, 0] = -10.0
    logits[3, 1] = 10.0            # ... except anchor 3, which is a person
    regression = np.zeros((num_anchors, 4), dtype=np.float32)

    dets = postprocess_detections(
        logits, regression, anchors,
        input_size=(320, 320), original_size=(640, 320),
        score_threshold=0.5, nms_iou=0.45, keep_labels=[1, 3],
    )
    assert len(dets) == 1
    assert int(dets.labels[0]) == 1
    assert dets.scores[0] > 0.99
    # original_size doubles the height only
    assert np.allclose(dets.boxes[0], [100.0, 200.0, 140.0, 360.0], atol=1e-3)


def test_postprocess_returns_empty_below_threshold():
    anchors = np.tile(np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32), (4, 1))
    logits = np.zeros((4, 91), dtype=np.float32)  # uniform -> p = 1/91 each
    dets = postprocess_detections(logits, np.zeros((4, 4), dtype=np.float32), anchors,
                                  score_threshold=0.5, keep_labels=[1])
    assert len(dets) == 0
    assert dets.boxes.shape == (0, 4)


def test_postprocess_rejects_batched_input():
    anchors = np.zeros((2, 4), dtype=np.float32)
    logits = np.zeros((2, 2, 91), dtype=np.float32)
    with pytest.raises(ValueError):
        postprocess_detections(logits, np.zeros((2, 2, 4), dtype=np.float32), anchors)


def test_merge_detections_deduplicates_across_tiles():
    a = Detections(np.array([[0, 0, 10, 10]], dtype=np.float32),
                   np.array([0.9], dtype=np.float32), np.array([1]))
    b = Detections(np.array([[1, 1, 11, 11]], dtype=np.float32),
                   np.array([0.8], dtype=np.float32), np.array([1]))
    c = Detections(np.array([[80, 80, 90, 90]], dtype=np.float32),
                   np.array([0.7], dtype=np.float32), np.array([3]))
    merged = merge_detections([a, b, c], nms_iou=0.5)
    assert len(merged) == 2
    assert sorted(int(x) for x in merged.labels) == [1, 3]


def test_merge_detections_handles_all_empty():
    assert len(merge_detections([Detections.empty(), Detections.empty()])) == 0


def test_suppress_cross_class_keeps_person_next_to_vehicle():
    """A car labelled twice collapses; an overlapping person survives."""
    boxes = np.array(
        [[0, 0, 100, 100], [2, 2, 98, 98], [0, 0, 100, 100]], dtype=np.float32
    )
    scores = np.array([0.7, 0.6, 0.55], dtype=np.float32)
    labels = np.array([8, 3, 1])  # truck, car, person
    out = suppress_cross_class_duplicates(Detections(boxes, scores, labels),
                                          group_labels=[2, 3, 4, 6, 8], iou_threshold=0.6)
    kept = sorted(int(x) for x in out.labels)
    assert kept == [1, 8]


def test_suppress_cross_class_is_a_noop_without_a_group():
    dets = Detections(np.array([[0, 0, 10, 10]], dtype=np.float32),
                      np.array([0.9], dtype=np.float32), np.array([1]))
    assert len(suppress_cross_class_duplicates(dets, group_labels=[])) == 1
