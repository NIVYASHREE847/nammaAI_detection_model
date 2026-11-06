"""Tests for zone geometry, the conflict rule and class counting."""

from __future__ import annotations

import numpy as np
import pytest

from safecross import coco_name_to_id
from safecross.counter import ClassCounter
from safecross.postprocess import Detections
from safecross.zones import ConflictRule, Zone, anchor_points, points_in_polygon

CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
VEHICLES = ["bicycle", "car", "motorcycle", "bus", "truck"]
SQUARE = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dtype=np.float32)


def make_dets(entries):
    """entries: list of (x0, y0, x1, y1, class_name, score)."""
    if not entries:
        return Detections.empty()
    boxes = np.array([e[:4] for e in entries], dtype=np.float32)
    labels = np.array([coco_name_to_id(e[4]) for e in entries], dtype=np.int64)
    scores = np.array([e[5] for e in entries], dtype=np.float32)
    return Detections(boxes, scores, labels)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_points_in_polygon_square():
    pts = np.array([[50, 50], [-1, 50], [101, 50], [50, -1], [50, 101]], dtype=np.float32)
    inside = points_in_polygon(pts, SQUARE)
    assert inside.tolist() == [True, False, False, False, False]


def test_points_in_polygon_concave_shape():
    """A C-shape: the notch must read as outside."""
    poly = np.array(
        [[0, 0], [100, 0], [100, 40], [40, 40], [40, 60], [100, 60], [100, 100], [0, 100]],
        dtype=np.float32,
    )
    pts = np.array([[20, 50], [70, 50], [70, 20], [70, 80]], dtype=np.float32)
    assert points_in_polygon(pts, poly).tolist() == [True, False, True, True]


def test_points_in_polygon_handles_horizontal_edges():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    pts = np.array([[5, 5], [5, 20]], dtype=np.float32)
    assert points_in_polygon(pts, poly).tolist() == [True, False]


def test_points_in_polygon_empty_input():
    assert points_in_polygon(np.zeros((0, 2)), SQUARE).shape == (0,)


def test_points_in_polygon_needs_three_vertices():
    with pytest.raises(ValueError):
        points_in_polygon(np.array([[1.0, 1.0]]), np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_anchor_point_modes():
    boxes = np.array([[10.0, 20.0, 30.0, 60.0]], dtype=np.float32)
    assert anchor_points(boxes, "bottom_center").tolist() == [[20.0, 60.0]]
    assert anchor_points(boxes, "center").tolist() == [[20.0, 40.0]]
    assert anchor_points(boxes, "top_center").tolist() == [[20.0, 20.0]]
    with pytest.raises(ValueError):
        anchor_points(boxes, "nope")


def test_bottom_center_excludes_a_tall_person_standing_outside():
    """A pedestrian on the kerb whose head overhangs the road is NOT in the road."""
    zone = Zone(np.array([[0, 80], [100, 80], [100, 100], [0, 100]], dtype=np.float32))
    # Box spans y 20..75: tall, overlaps the zone's airspace, feet at y=75 outside.
    boxes = np.array([[40.0, 20.0, 60.0, 75.0]], dtype=np.float32)
    assert zone.contains_boxes(boxes).tolist() == [False]
    # Same person two steps later, feet at y=85 -> inside.
    boxes[0, 3] = 85.0
    assert zone.contains_boxes(boxes).tolist() == [True]


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #
def test_counter_counts_per_class_and_vehicle_total():
    counter = ClassCounter(CLASSES, VEHICLES)
    dets = make_dets([
        (0, 0, 1, 1, "person", 0.9),
        (0, 0, 1, 1, "person", 0.8),
        (0, 0, 1, 1, "car", 0.9),
        (0, 0, 1, 1, "bus", 0.7),
        (0, 0, 1, 1, "bicycle", 0.6),
    ])
    counts = counter.count(dets)
    assert counts.per_class["person"] == 2
    assert counts.per_class["car"] == 1
    assert counts.per_class["truck"] == 0
    assert counts.vehicle_total == 3  # car + bus + bicycle
    assert counts.total == 5


def test_counter_on_empty_detections():
    counter = ClassCounter(CLASSES, VEHICLES)
    counts = counter.count(Detections.empty())
    assert counts.vehicle_total == 0
    assert all(v == 0 for v in counts.per_class.values())


def test_csv_row_order_matches_csv_columns():
    counter = ClassCounter(CLASSES, VEHICLES)
    dets = make_dets([(0, 0, 1, 1, "truck", 0.9)])
    row = counter.count(dets).as_row(counter.class_names)
    assert len(row) == len(counter.csv_columns)
    assert row[counter.csv_columns.index("truck")] == 1
    assert row[counter.csv_columns.index("vehicle_total")] == 1


def test_counter_rejects_unknown_class_and_stray_vehicle():
    with pytest.raises(ValueError):
        ClassCounter(["person", "unicorn"], ["unicorn"])
    with pytest.raises(ValueError):
        ClassCounter(["person"], ["car"])


def test_person_vehicle_split():
    counter = ClassCounter(CLASSES, VEHICLES)
    dets = make_dets([
        (0, 0, 1, 1, "person", 0.9),
        (0, 0, 1, 1, "truck", 0.9),
        (0, 0, 1, 1, "motorcycle", 0.9),
    ])
    assert counter.person_vehicle_split(dets) == (1, 2)


# --------------------------------------------------------------------------- #
# zone + conflict rule
# --------------------------------------------------------------------------- #
def test_zone_evaluate_flags_a_conflict():
    counter = ClassCounter(CLASSES, VEHICLES)
    zone = Zone(SQUARE, min_persons=1, min_vehicles=1)
    dets = make_dets([
        (10, 10, 20, 50, "person", 0.9),     # feet at y=50 -> inside
        (10, 10, 60, 90, "car", 0.9),        # inside
        (200, 200, 260, 290, "car", 0.9),    # far away -> outside
    ])
    result = zone.evaluate(dets, counter)
    assert result.persons_in_zone == 1
    assert result.vehicles_in_zone == 1
    assert result.conflict is True
    assert result.mask.tolist() == [True, True, False]


def test_conflict_needs_both_sides():
    counter = ClassCounter(CLASSES, VEHICLES)
    zone = Zone(SQUARE, min_persons=1, min_vehicles=1)
    only_people = make_dets([(10, 10, 20, 50, "person", 0.9)])
    assert zone.evaluate(only_people, counter).conflict is False
    only_cars = make_dets([(10, 10, 20, 50, "car", 0.9)])
    assert zone.evaluate(only_cars, counter).conflict is False


def test_conflict_thresholds_are_honoured():
    counter = ClassCounter(CLASSES, VEHICLES)
    zone = Zone(SQUARE, min_persons=3, min_vehicles=2)
    dets = make_dets([
        (10, 10, 20, 50, "person", 0.9),
        (20, 10, 30, 50, "person", 0.9),
        (30, 10, 40, 50, "car", 0.9),
        (40, 10, 50, 50, "car", 0.9),
    ])
    result = zone.evaluate(dets, counter)
    assert (result.persons_in_zone, result.vehicles_in_zone) == (2, 2)
    assert result.conflict is False  # 2 persons < min_persons=3


def test_disabled_zone_counts_the_whole_frame():
    counter = ClassCounter(CLASSES, VEHICLES)
    zone = Zone(np.zeros((0, 2), dtype=np.float32), enabled=False)
    dets = make_dets([(900, 900, 950, 990, "person", 0.9),
                      (900, 900, 950, 990, "car", 0.9)])
    result = zone.evaluate(dets, counter)
    assert result.mask.all()
    assert result.conflict is True


def test_zone_from_config_scales_normalized_vertices():
    cfg = {
        "zone": {
            "enabled": True,
            "polygon_norm": [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]],
            "anchor_point": "bottom_center",
        },
        "conflict": {"min_persons": 2, "min_vehicles": 1},
    }
    zone = Zone.from_config(cfg, width=1280, height=720)
    assert zone.polygon_int.tolist() == [[0, 360], [1280, 360], [1280, 720], [0, 720]]
    assert (zone.min_persons, zone.min_vehicles) == (2, 1)


def test_zone_rejects_degenerate_polygon_when_enabled():
    with pytest.raises(ValueError):
        Zone(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), enabled=True)


# --------------------------------------------------------------------------- #
# minimum-duration conflict rule
# --------------------------------------------------------------------------- #
def _run_rule(rule, sequence):
    """Feed (persons, vehicles) pairs and collect the flags."""
    return [rule.update(i, i / 30.0, p, v)[0] for i, (p, v) in enumerate(sequence)]


def test_duration_one_behaves_like_a_plain_threshold():
    rule = ConflictRule(min_persons=1, min_vehicles=1, min_consecutive_frames=1)
    flags = _run_rule(rule, [(1, 1), (0, 1), (1, 1)])
    assert flags == [True, False, True]


def test_duration_rule_suppresses_short_bursts():
    rule = ConflictRule(min_persons=1, min_vehicles=1, min_consecutive_frames=3)
    # condition true for only 2 frames, then false
    flags = _run_rule(rule, [(1, 1), (1, 1), (0, 0), (1, 1)])
    assert flags == [False, False, False, False]
    assert rule.finish() == []


def test_duration_rule_fires_once_sustained_and_stays_on():
    rule = ConflictRule(min_persons=1, min_vehicles=1, min_consecutive_frames=3)
    flags = _run_rule(rule, [(1, 1)] * 5)
    assert flags == [False, False, True, True, True]
    episodes = rule.finish()
    assert len(episodes) == 1
    assert episodes[0]["start_frame"] == 0
    assert episodes[0]["end_frame"] == 4
    assert episodes[0]["frames"] == 5


def test_streak_resets_on_a_single_gap():
    rule = ConflictRule(min_persons=1, min_vehicles=1, min_consecutive_frames=3)
    flags = _run_rule(rule, [(1, 1), (1, 1), (0, 1), (1, 1), (1, 1), (1, 1)])
    assert flags == [False, False, False, False, False, True]
    assert len(rule.finish()) == 1


def test_episode_records_peak_counts():
    rule = ConflictRule(min_persons=2, min_vehicles=1, min_consecutive_frames=2)
    _run_rule(rule, [(2, 1), (5, 3), (3, 2)])
    episode = rule.finish()[0]
    assert episode["max_persons"] == 5
    assert episode["max_vehicles"] == 3


def test_thresholds_gate_before_duration():
    rule = ConflictRule(min_persons=2, min_vehicles=1, min_consecutive_frames=2)
    # one person is never enough, however long it lasts
    assert _run_rule(rule, [(1, 5)] * 10) == [False] * 10


def test_zone_evaluate_threads_the_rule_through():
    counter = ClassCounter(CLASSES, VEHICLES)
    zone = Zone(SQUARE)
    rule = ConflictRule(min_persons=1, min_vehicles=1, min_consecutive_frames=2)
    dets = make_dets([(10, 10, 20, 50, "person", 0.9), (10, 10, 60, 90, "car", 0.9)])
    first = zone.evaluate(dets, counter, rule, frame_idx=0, timestamp_s=0.0)
    second = zone.evaluate(dets, counter, rule, frame_idx=1, timestamp_s=0.03)
    assert first.condition is True and first.conflict is False and first.streak == 1
    assert second.conflict is True and second.streak == 2


def test_rule_from_config():
    rule = ConflictRule.from_config(
        {"conflict": {"min_persons": 3, "min_vehicles": 2, "min_consecutive_frames": 7}}
    )
    assert (rule.min_persons, rule.min_vehicles, rule.min_consecutive_frames) == (3, 2, 7)
    assert "sustained for >= 7" in rule.describe()
