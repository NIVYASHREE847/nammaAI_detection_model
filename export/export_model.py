"""Export ssdlite320_mobilenet_v3_large to ExecuTorch .pte (FP32 and INT8).

No training happens anywhere in SafeCross. This script takes the pretrained
COCO weights, strips everything that is not a static convolutional graph, and
lowers that graph to an ExecuTorch program for the XNNPACK CPU backend.

What gets exported
------------------
Only ``backbone + SSD head``, emitting raw ``(cls_logits, bbox_regression)``.
The input transform, anchor generation, softmax, score threshold, NMS and the
box rescale all live in NumPy (``safecross/preprocess.py`` and
``safecross/postprocess.py``). Reason: NMS output is data-dependent, so it
would force dynamic shapes into the program and block delegation. A pure
static-shape CNN is what runs well on a CPU backend, on a laptop or a Pi.

Alongside the .pte files this writes
  models/anchors.npy      -- the 3234x4 anchor table, produced by torchvision's
                             own DefaultBoxGenerator, so no anchor maths is
                             re-derived by hand anywhere in this repo
  models/model_meta.json  -- input size, mean/std, box-coder weights, class
                             table, and torchvision's own default thresholds

INT8
----
PT2E post-training static quantization with ``XNNPACKQuantizer``
(per-channel symmetric weights, per-tensor activations), calibrated on frames
sampled evenly from the input video, so the activation ranges come from the
footage the model will actually see.

Parity
------
After each export the program is run through the ExecuTorch runtime on real
frames and compared against torchvision eager mode, both at raw-tensor level
(max abs difference on logits and regressions) and at detection level
(box IoU / score / label agreement after the NumPy post-processing). Results go
to ``models/parity_<precision>.json`` and are printed. INT8 is expected to
differ; the numbers are reported rather than hidden.

Usage
-----
    python export/export_model.py                        # FP32 + INT8
    python export/export_model.py --precision fp32
    python export/export_model.py --calib-frames 100 --parity-frames 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ensure_flatc() -> str | None:
    """Point ExecuTorch at ``flatc`` on Windows.

    The XNNPACK serializer shells out to ``flatc``. ExecuTorch looks for a
    bundled resource literally named ``flatc``, but the win_amd64 wheel ships
    it as ``flatc.exe``, so the lookup misses and the export dies with
    ``FileNotFoundError: [WinError 2]``. ExecuTorch's own documented escape
    hatch is the ``FLATC_EXECUTABLE`` environment variable, so set it to the
    binary that is already installed. No-op on Linux/macOS and whenever the
    variable is already set, so this costs nothing on a Raspberry Pi.
    """
    if os.environ.get("FLATC_EXECUTABLE"):
        return os.environ["FLATC_EXECUTABLE"]
    if os.name != "nt":
        return None
    candidates = [
        Path(sys.prefix) / "Scripts" / "flatc.exe",
        Path(sys.executable).parent / "flatc.exe",
    ]
    try:
        import executorch  # noqa: F401  (import only to locate the package dir)

        candidates.insert(0, Path(executorch.__file__).parent / "data" / "bin" / "flatc.exe")
    except Exception:
        pass
    found = shutil.which("flatc") or shutil.which("flatc.exe")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            os.environ["FLATC_EXECUTABLE"] = str(candidate)
            return str(candidate)
    return None


from safecross import load_config, resolve  # noqa: E402
from safecross.counter import ClassCounter  # noqa: E402
from safecross.postprocess import postprocess_detections  # noqa: E402
from safecross.preprocess import preprocess_frame  # noqa: E402
from safecross.runtime import build_raw_ssd  # noqa: E402
from safecross.video import sample_frames  # noqa: E402


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_fp32(module, example_input, out_path: Path) -> dict:
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    t0 = time.perf_counter()
    with torch.no_grad():
        exported = torch.export.export(module, (example_input,), strict=True)
    lowered = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    program = lowered.to_executorch()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(program.buffer)
    elapsed = time.perf_counter() - t0
    return {
        "precision": "fp32",
        "path": str(out_path),
        "size_mb": out_path.stat().st_size / (1024 * 1024),
        "export_seconds": elapsed,
    }


def export_int8(module, example_input, calib_tensors, out_path: Path) -> dict:
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from executorch.exir import to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    t0 = time.perf_counter()
    with torch.no_grad():
        captured = torch.export.export(module, (example_input,), strict=True).module()

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        prepared = prepare_pt2e(captured, quantizer)

        # Calibration: observers record activation ranges from real frames.
        for tensor in calib_tensors:
            prepared(torch.from_numpy(tensor))

        quantized = convert_pt2e(prepared)
        exported = torch.export.export(quantized, (example_input,), strict=True)

    lowered = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    program = lowered.to_executorch()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(program.buffer)
    elapsed = time.perf_counter() - t0
    return {
        "precision": "int8",
        "path": str(out_path),
        "size_mb": out_path.stat().st_size / (1024 * 1024),
        "export_seconds": elapsed,
        "calibration_frames": len(calib_tensors),
        "quantization": "PT2E static, XNNPACKQuantizer, per-channel symmetric weights",
    }


# --------------------------------------------------------------------------- #
# parity
# --------------------------------------------------------------------------- #
def _match_detections(ref, got, iou_threshold: float = 0.5):
    """Greedily match two detection sets by label + IoU. Returns match stats."""
    from safecross.postprocess import Detections  # local import keeps module list tidy

    assert isinstance(ref, Detections) and isinstance(got, Detections)
    used = np.zeros(len(got), dtype=bool)
    ious: list[float] = []
    score_deltas: list[float] = []
    matched = 0
    for i in range(len(ref)):
        best_j, best_iou = -1, 0.0
        for j in range(len(got)):
            if used[j] or int(got.labels[j]) != int(ref.labels[i]):
                continue
            ax0, ay0, ax1, ay1 = ref.boxes[i]
            bx0, by0, bx1, by1 = got.boxes[j]
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
            area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
            area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
            union = area_a + area_b - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_threshold:
            used[best_j] = True
            matched += 1
            ious.append(best_iou)
            score_deltas.append(abs(float(ref.scores[i]) - float(got.scores[best_j])))
    return {
        "reference_detections": len(ref),
        "runtime_detections": len(got),
        "matched": matched,
        "unmatched_reference": len(ref) - matched,
        "unmatched_runtime": len(got) - matched,
        "mean_iou_of_matches": float(np.mean(ious)) if ious else None,
        "max_score_delta_of_matches": float(np.max(score_deltas)) if score_deltas else None,
    }


def parity_check(
    pte_path: Path,
    precision: str,
    eager_module,
    anchors: np.ndarray,
    meta: dict,
    tensors: list[np.ndarray],
    frame_sizes: list[tuple[int, int]],
    det_cfg: dict,
    keep_labels: list[int],
) -> dict:
    """Compare .pte outputs against torchvision eager outputs on the same frames."""
    import torch

    from safecross.runtime import ExecuTorchRuntime

    runtime = ExecuTorchRuntime(pte_path, precision=precision)

    logit_err: list[float] = []
    bbox_err: list[float] = []
    det_stats: list[dict] = []

    for tensor, size in zip(tensors, frame_sizes):
        with torch.no_grad():
            ref_logits, ref_bbox = eager_module(torch.from_numpy(tensor))
        ref_logits = ref_logits.numpy()
        ref_bbox = ref_bbox.numpy()
        got_logits, got_bbox = runtime(tensor)

        logit_err.append(float(np.max(np.abs(ref_logits - got_logits))))
        bbox_err.append(float(np.max(np.abs(ref_bbox - got_bbox))))

        kwargs = dict(
            input_size=tuple(meta["input_size"]),
            original_size=size,
            score_threshold=float(det_cfg["score_threshold"]),
            nms_iou=float(det_cfg["nms_iou"]),
            topk_candidates=int(det_cfg["topk_candidates"]),
            detections_per_img=int(det_cfg["detections_per_img"]),
            keep_labels=keep_labels,
            box_weights=meta["box_weights"],
        )
        ref_dets = postprocess_detections(ref_logits, ref_bbox, anchors, **kwargs)
        got_dets = postprocess_detections(got_logits, got_bbox, anchors, **kwargs)
        det_stats.append(_match_detections(ref_dets, got_dets))

    totals = {
        key: int(sum(s[key] for s in det_stats))
        for key in ("reference_detections", "runtime_detections", "matched",
                    "unmatched_reference", "unmatched_runtime")
    }
    recall = totals["matched"] / totals["reference_detections"] if totals["reference_detections"] else 1.0
    return {
        "precision": precision,
        "pte": str(pte_path),
        "frames": len(tensors),
        "raw_tensor": {
            "max_abs_cls_logit_diff": max(logit_err) if logit_err else 0.0,
            "mean_max_abs_cls_logit_diff": float(np.mean(logit_err)) if logit_err else 0.0,
            "max_abs_bbox_regression_diff": max(bbox_err) if bbox_err else 0.0,
            "mean_max_abs_bbox_regression_diff": float(np.mean(bbox_err)) if bbox_err else 0.0,
        },
        "detections_vs_eager": {
            **totals,
            "match_rate_vs_eager": round(float(recall), 4),
            "per_frame": det_stats,
        },
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--calib-source", default="data/test_video.mp4",
                        help="video or image folder used to calibrate INT8")
    parser.add_argument("--calib-frames", type=int, default=64,
                        help="frames sampled evenly for INT8 calibration (50-100 is plenty)")
    parser.add_argument("--parity-frames", type=int, default=6,
                        help="frames used for the .pte vs eager comparison")
    parser.add_argument("--precision", choices=["fp32", "int8", "both"], default="both")
    parser.add_argument("--outdir", default="models")
    args = parser.parse_args(argv)

    flatc = _ensure_flatc()
    if flatc:
        print(f"      using flatc: {flatc}")

    import torch

    torch.manual_seed(0)
    cfg = load_config(args.config)
    outdir = resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/5] building pretrained model (no training, COCO weights)")
    module, anchors, meta = build_raw_ssd()
    counter = ClassCounter.from_config(cfg, meta["categories"])
    np.save(outdir / "anchors.npy", anchors)
    with open(outdir / "model_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"      anchors {anchors.shape} -> {outdir / 'anchors.npy'}")
    print(f"      {meta['num_classes']} classes, input {meta['input_size']}, "
          f"mean={meta['image_mean']} std={meta['image_std']}")

    source = resolve(args.calib_source)
    if not source.exists():
        raise SystemExit(f"calibration source not found: {source}")

    print(f"[2/5] sampling frames from {source}")
    n_needed = max(args.calib_frames, args.parity_frames)
    frames = sample_frames(source, n_needed)
    size = tuple(meta["input_size"])
    tensors = [
        preprocess_frame(f.image, size=size, mean=meta["image_mean"], std=meta["image_std"])
        for f in frames
    ]
    frame_sizes = [(f.image.shape[0], f.image.shape[1]) for f in frames]
    calib_tensors = tensors[: args.calib_frames]
    # Parity frames spread across the clip rather than only its opening.
    stride = max(1, len(tensors) // max(1, args.parity_frames))
    parity_idx = list(range(0, len(tensors), stride))[: args.parity_frames]
    parity_tensors = [tensors[i] for i in parity_idx]
    parity_sizes = [frame_sizes[i] for i in parity_idx]
    print(f"      {len(tensors)} frames, {len(calib_tensors)} for calibration, "
          f"{len(parity_tensors)} for parity")

    example = torch.from_numpy(tensors[0])
    results: dict[str, dict] = {}

    want = ["fp32", "int8"] if args.precision == "both" else [args.precision]

    if "fp32" in want:
        print("[3/5] exporting FP32 -> ExecuTorch/XNNPACK")
        fp32_path = resolve(cfg["model"]["fp32_pte"])
        results["fp32"] = export_fp32(module, example, fp32_path)
        print(f"      {fp32_path}  {results['fp32']['size_mb']:.2f} MB "
              f"({results['fp32']['export_seconds']:.1f}s)")

    if "int8" in want:
        print(f"[4/5] exporting INT8 (PT2E, {len(calib_tensors)} calibration frames)")
        int8_path = resolve(cfg["model"]["int8_pte"])
        results["int8"] = export_int8(module, example, calib_tensors, int8_path)
        print(f"      {int8_path}  {results['int8']['size_mb']:.2f} MB "
              f"({results['int8']['export_seconds']:.1f}s)")

    print("[5/5] parity check against torchvision eager mode")
    for precision in want:
        pte = resolve(cfg["model"][f"{precision}_pte"])
        report = parity_check(
            pte, precision, module, anchors, meta, parity_tensors, parity_sizes,
            cfg["detection"], counter.keep_labels,
        )
        results[precision]["parity"] = report
        with open(outdir / f"parity_{precision}.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        raw = report["raw_tensor"]
        det = report["detections_vs_eager"]
        print(f"      {precision}: max|cls_logit diff|={raw['max_abs_cls_logit_diff']:.3e}  "
              f"max|bbox diff|={raw['max_abs_bbox_regression_diff']:.3e}")
        print(f"      {precision}: detections eager={det['reference_detections']} "
              f"pte={det['runtime_detections']} matched={det['matched']} "
              f"(match rate {det['match_rate_vs_eager']:.3f})")

    with open(outdir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {outdir / 'export_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
