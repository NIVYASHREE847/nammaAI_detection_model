"""Integration tests: the exported .pte must agree with torchvision eager mode.

Every test here skips cleanly when the artefacts are missing, so ``pytest``
passes on a fresh clone before ``export/export_model.py`` has been run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from safecross import ROOT, load_config
from safecross.counter import ClassCounter
from safecross.postprocess import postprocess_detections
from safecross.preprocess import preprocess_frame

MODELS = ROOT / "models"
META_PATH = MODELS / "model_meta.json"
ANCHORS_PATH = MODELS / "anchors.npy"
VIDEO = ROOT / "data" / "test_video.mp4"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _require(path: Path, hint: str):
    if not path.exists():
        pytest.skip(f"{path.name} missing -- run {hint}")


@pytest.fixture(scope="module")
def meta():
    _require(META_PATH, "python export/export_model.py")
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def anchors():
    _require(ANCHORS_PATH, "python export/export_model.py")
    return np.load(ANCHORS_PATH)


@pytest.fixture(scope="module")
def frames():
    _require(VIDEO, "see data/SOURCE.md")
    from safecross.video import sample_frames

    return sample_frames(VIDEO, 3)


def test_meta_describes_the_expected_model(meta):
    assert meta["model"] == "ssdlite320_mobilenet_v3_large"
    assert meta["input_size"] == [320, 320]
    assert meta["image_mean"] == [0.5, 0.5, 0.5]
    assert meta["image_std"] == [0.5, 0.5, 0.5]
    assert meta["box_weights"] == [10.0, 10.0, 5.0, 5.0]
    assert meta["num_classes"] == 91
    assert meta["categories"][1] == "person"


def test_anchor_table_shape_matches_meta(meta, anchors):
    assert anchors.shape == (meta["num_anchors"], 4)
    assert anchors.shape[0] == 3234
    # SSD anchors are allowed outside the image; decoding clips afterwards.
    assert anchors[:, 2].max() > 320


@pytest.mark.parametrize("precision", ["fp32", "int8"])
def test_pte_exists_and_is_smaller_than_the_eager_parameters(precision):
    pte = MODELS / f"ssdlite320_{precision}.pte"
    _require(pte, "python export/export_model.py")
    size_mb = pte.stat().st_size / (1024 * 1024)
    assert 0.5 < size_mb < 60


def test_int8_is_substantially_smaller_than_fp32():
    fp32, int8 = MODELS / "ssdlite320_fp32.pte", MODELS / "ssdlite320_int8.pte"
    _require(fp32, "python export/export_model.py")
    _require(int8, "python export/export_model.py")
    assert int8.stat().st_size < 0.6 * fp32.stat().st_size


def test_fp32_pte_matches_eager_outputs(meta, anchors, frames):
    """The FP32 program is a lowering, not an approximation: it must match closely."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pte = MODELS / "ssdlite320_fp32.pte"
    _require(pte, "python export/export_model.py")

    from safecross.runtime import EagerRuntime, ExecuTorchRuntime

    eager = EagerRuntime()
    executorch = ExecuTorchRuntime(pte, precision="fp32")
    try:
        for frame in frames:
            tensor = preprocess_frame(frame.image, size=tuple(meta["input_size"]),
                                      mean=meta["image_mean"], std=meta["image_std"])
            ref_logits, ref_bbox = eager(tensor)
            got_logits, got_bbox = executorch(tensor)
            assert got_logits.shape == ref_logits.shape
            assert np.max(np.abs(ref_logits - got_logits)) < 1e-2
            assert np.max(np.abs(ref_bbox - got_bbox)) < 1e-2
    finally:
        eager.close()
        executorch.close()
    del torch


def test_fp32_detections_match_eager_detections(meta, anchors, frames):
    """Same frames, same thresholds -> same boxes, labels and scores."""
    pytest.importorskip("torchvision")
    pte = MODELS / "ssdlite320_fp32.pte"
    _require(pte, "python export/export_model.py")

    from safecross.runtime import EagerRuntime, ExecuTorchRuntime

    cfg = load_config(ROOT / "config.yaml")
    counter = ClassCounter.from_config(cfg, meta["categories"])
    kwargs = dict(
        input_size=tuple(meta["input_size"]),
        score_threshold=float(cfg["detection"]["score_threshold"]),
        nms_iou=float(cfg["detection"]["nms_iou"]),
        keep_labels=counter.keep_labels,
        box_weights=meta["box_weights"],
    )

    eager = EagerRuntime()
    executorch = ExecuTorchRuntime(pte, precision="fp32")
    try:
        for frame in frames:
            size = (frame.image.shape[0], frame.image.shape[1])
            tensor = preprocess_frame(frame.image, size=tuple(meta["input_size"]),
                                      mean=meta["image_mean"], std=meta["image_std"])
            ref = postprocess_detections(*eager(tensor), anchors, original_size=size, **kwargs)
            got = postprocess_detections(*executorch(tensor), anchors, original_size=size, **kwargs)
            assert len(ref) == len(got)
            if len(ref):
                assert np.array_equal(ref.labels, got.labels)
                assert np.max(np.abs(ref.boxes - got.boxes)) < 1.0  # sub-pixel
                assert np.max(np.abs(ref.scores - got.scores)) < 1e-2
    finally:
        eager.close()
        executorch.close()


def test_recorded_parity_report_is_within_expectations():
    """Guards the numbers the report quotes, so a bad re-export is caught."""
    fp32_report = MODELS / "parity_fp32.json"
    _require(fp32_report, "python export/export_model.py")
    data = json.loads(fp32_report.read_text(encoding="utf-8"))
    assert data["raw_tensor"]["max_abs_cls_logit_diff"] < 1e-2
    assert data["detections_vs_eager"]["match_rate_vs_eager"] >= 0.99

    int8_report = MODELS / "parity_int8.json"
    if int8_report.exists():
        int8 = json.loads(int8_report.read_text(encoding="utf-8"))
        # INT8 is expected to drift. This only asserts it has not collapsed.
        assert int8["detections_vs_eager"]["match_rate_vs_eager"] >= 0.5
