"""Measure counting accuracy against a hand-labelled ground truth, and sweep thresholds.

Workflow
--------
1. Export frames to count by hand (default 10, spread evenly across the clip)::

       python scripts/eval_counts.py --export-frames 10

   This writes ``outputs/gt_frames/*.jpg`` and a ready-to-fill
   ``outputs/gt_template.csv`` with the right ``frame_idx`` values already in
   it. Open the images, count, and type the numbers in.

2. Score the model against your counts and sweep the score threshold::

       python scripts/eval_counts.py --gt outputs/gt_template.csv
       python scripts/eval_counts.py --gt my_counts.csv --tile

Ground-truth CSV format
-----------------------
``frame_idx,person,vehicle`` -- one row per labelled frame. ``vehicle`` is the
total of every vehicle class together (car + motorcycle + bus + truck +
bicycle), because telling a van from a truck by eye is unreliable and the
pipeline reports ``vehicle_total`` anyway. Extra columns are ignored, so the
file ``--export-frames`` writes can be edited in a spreadsheet.

Metrics
-------
For each class and threshold, over the labelled frames:

* **MAE**   -- mean absolute error in objects per frame
* **bias**  -- mean signed error (negative = the model under-counts)
* **RMSE**  -- penalises the occasional badly-wrong frame
* **accuracy** -- ``1 - MAE / mean(ground truth)``, clipped at 0. A model that
  is on average off by 1 object where the truth averages 10 scores 0.90. It is
  a count-level accuracy, NOT detection precision/recall; no box matching is
  involved, so a frame can score perfectly with the right number of boxes in
  the wrong places.
* **exact** -- fraction of frames where the count is exactly right

The model runs ONCE per frame; the sweep re-runs only the NumPy
post-processing on the cached raw outputs, so sweeping 9 thresholds costs
almost nothing extra.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safecross import load_config, load_meta, resolve  # noqa: E402
from safecross.counter import ClassCounter  # noqa: E402
from safecross.postprocess import (  # noqa: E402
    Detections,
    merge_detections,
    postprocess_detections,
    suppress_cross_class_duplicates,
)
from safecross.preprocess import preprocess_crop, preprocess_frame, tile_rects  # noqa: E402
from safecross.runtime import build_runtime, load_anchors  # noqa: E402
from safecross.video import iter_frames, probe_source, sample_frames  # noqa: E402

DEFAULT_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# --------------------------------------------------------------------------- #
# ground-truth template
# --------------------------------------------------------------------------- #
def export_frames(source: Path, count: int, outdir: Path) -> Path:
    """Write ``count`` evenly spaced frames plus a GT template to fill in."""
    frames_dir = outdir / "gt_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = sample_frames(source, count)
    if not frames:
        raise SystemExit(f"no frames read from {source}")

    for frame in frames:
        cv2.imwrite(str(frames_dir / f"frame_{frame.index:06d}.jpg"), frame.image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    template = outdir / "gt_template.csv"
    with open(template, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "person", "vehicle", "notes"])
        for frame in frames:
            writer.writerow([frame.index, "", "", ""])

    print(f"wrote {len(frames)} frames to {frames_dir}")
    print(f"wrote ground-truth template to {template}")
    print()
    print("Now count by hand:")
    print(f"  1. open the images in {frames_dir}")
    print("  2. for each frame, count people and count vehicles "
          "(car + bus + truck + motorcycle + bicycle together)")
    print(f"  3. fill the person and vehicle columns in {template}")
    print(f'  4. run:  python scripts/eval_counts.py --gt "{template}"')
    print()
    print("Count every object a human can identify, including partly occluded "
          "ones and ones at the far end of the street -- that is the number the "
          "detector is being measured against.")
    return template


def read_ground_truth(path: Path) -> dict[int, dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            if row.get("frame_idx") in (None, ""):
                continue
            person, vehicle = row.get("person", ""), row.get("vehicle", "")
            if str(person).strip() == "" or str(vehicle).strip() == "":
                continue  # not filled in yet -- skip rather than guess a zero
            try:
                rows[int(row["frame_idx"])] = {
                    "person": int(float(person)),
                    "vehicle": int(float(vehicle)),
                }
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_no}: could not parse row {row}: {exc}") from exc
    if not rows:
        raise SystemExit(
            f"{path} has no filled-in rows. Fill the person and vehicle columns first "
            "(see --export-frames)."
        )
    return rows


# --------------------------------------------------------------------------- #
# inference with cached raw outputs
# --------------------------------------------------------------------------- #
def collect_raw(source: Path, wanted: set[int], runtime, meta: dict, tile_cfg: dict | None):
    """Run the model once per wanted frame, keeping raw head outputs for the sweep."""
    size = tuple(meta["input_size"])
    mean, std = meta["image_mean"], meta["image_std"]
    cached: dict[int, list[tuple]] = {}

    for frame in iter_frames(source):
        if frame.index not in wanted:
            continue
        height, width = frame.image.shape[:2]
        entries: list[tuple] = []
        if tile_cfg is None:
            nchw = preprocess_frame(frame.image, size=size, mean=mean, std=std)
            cls_logits, bbox = runtime(nchw)
            entries.append(((0, 0), (height, width), cls_logits, bbox))
        else:
            rects = tile_rects(
                height, width,
                grid=tuple(tile_cfg.get("tile_grid", (2, 2))),
                overlap=float(tile_cfg.get("tile_overlap", 0.15)),
                include_full_frame=bool(tile_cfg.get("tile_include_full_frame", True)),
            )
            for rect in rects:
                nchw = preprocess_crop(frame.image, rect, size=size, mean=mean, std=std)
                cls_logits, bbox = runtime(nchw)
                entries.append(((rect.x0, rect.y0), (rect.height, rect.width), cls_logits, bbox))
        cached[frame.index] = entries
        if len(cached) == len(wanted):
            break

    missing = sorted(wanted - set(cached))
    if missing:
        raise SystemExit(f"these ground-truth frames are not in {source}: {missing}")
    return cached


def detections_at(entries, anchors, det_kwargs, dedup) -> Detections:
    """Re-run only the NumPy post-processing for one threshold setting."""
    parts: list[Detections] = []
    for (ox, oy), orig_size, cls_logits, bbox in entries:
        dets = postprocess_detections(
            cls_logits, bbox, anchors, original_size=orig_size, **det_kwargs
        )
        if len(dets):
            boxes = dets.boxes.copy()
            boxes[:, [0, 2]] += ox
            boxes[:, [1, 3]] += oy
            parts.append(Detections(boxes, dets.scores, dets.labels))
    merged = (
        parts[0] if len(parts) == 1
        else merge_detections(parts, nms_iou=float(det_kwargs["nms_iou"]),
                              detections_per_img=int(det_kwargs["detections_per_img"]))
    )
    if dedup:
        merged = suppress_cross_class_duplicates(merged, dedup[0], dedup[1])
    return merged


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def score(pred: list[int], truth: list[int]) -> dict[str, float]:
    pred_a = np.asarray(pred, dtype=np.float64)
    truth_a = np.asarray(truth, dtype=np.float64)
    error = pred_a - truth_a
    mae = float(np.mean(np.abs(error)))
    mean_truth = float(np.mean(truth_a))
    return {
        "mae": mae,
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "accuracy": float(max(0.0, 1.0 - mae / mean_truth)) if mean_truth > 0 else float("nan"),
        "exact_frame_rate": float(np.mean(error == 0)),
        "mean_pred": float(np.mean(pred_a)),
        "mean_truth": mean_truth,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/test_video.mp4")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--export-frames", type=int, default=None, metavar="N",
                        help="write N frames + a GT template, then exit")
    parser.add_argument("--gt", default=None, help="filled-in ground-truth CSV")
    parser.add_argument("--model", choices=["fp32", "int8", "eager"], default="fp32")
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--sweep", default=None,
                        help="comma separated score thresholds (default 0.30..0.70)")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    outdir = resolve(args.outdir or cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    source = resolve(args.source)

    if args.export_frames is not None:
        export_frames(source, args.export_frames, outdir)
        return 0
    if not args.gt:
        parser.error("pass --gt <csv> to evaluate, or --export-frames N to make a template")

    gt = read_ground_truth(resolve(args.gt))
    meta = load_meta(resolve(cfg["model"]["meta"]))
    anchors = load_anchors(resolve(cfg["model"]["anchors"]))
    counter = ClassCounter.from_config(cfg, meta["categories"])
    dcfg = cfg["detection"]
    pcfg = cfg["pipeline"]
    tile = bool(args.tile or pcfg.get("tile", False))
    dedup = (
        (sorted(counter.vehicle_ids), float(dcfg.get("cross_class_iou", 0.6)))
        if dcfg.get("merge_duplicate_vehicles", True)
        else None
    )
    thresholds = (
        [float(x) for x in args.sweep.split(",")] if args.sweep else list(DEFAULT_SWEEP)
    )

    info = probe_source(source)
    print(f"source     : {info.describe()}")
    print(f"model      : {args.model}{' +tile' if tile else ''}")
    print(f"ground truth: {len(gt)} labelled frames from {args.gt}")
    print(f"sweep      : {thresholds}")
    print()

    runtime = build_runtime(args.model, cfg)
    try:
        cached = collect_raw(source, set(gt), runtime, meta, pcfg if tile else None)
    finally:
        runtime.close()

    frame_ids = sorted(gt)
    truth_person = [gt[i]["person"] for i in frame_ids]
    truth_vehicle = [gt[i]["vehicle"] for i in frame_ids]

    results = []
    for threshold in thresholds:
        det_kwargs = dict(
            input_size=tuple(meta["input_size"]),
            score_threshold=threshold,
            nms_iou=float(dcfg["nms_iou"]),
            topk_candidates=int(dcfg["topk_candidates"]),
            detections_per_img=int(dcfg["detections_per_img"]),
            keep_labels=counter.keep_labels,
            box_weights=meta["box_weights"],
        )
        pred_person, pred_vehicle, per_frame = [], [], []
        for frame_id in frame_ids:
            dets = detections_at(cached[frame_id], anchors, det_kwargs, dedup)
            counts = counter.count(dets)
            pred_person.append(counts.per_class["person"])
            pred_vehicle.append(counts.vehicle_total)
            per_frame.append({
                "frame_idx": frame_id,
                "person_pred": counts.per_class["person"],
                "person_gt": gt[frame_id]["person"],
                "vehicle_pred": counts.vehicle_total,
                "vehicle_gt": gt[frame_id]["vehicle"],
            })
        person = score(pred_person, truth_person)
        vehicle = score(pred_vehicle, truth_vehicle)
        results.append({
            "score_threshold": threshold,
            "person": person,
            "vehicle": vehicle,
            "combined_mae": (person["mae"] + vehicle["mae"]) / 2.0,
            "per_frame": per_frame,
        })

    best = min(results, key=lambda r: r["combined_mae"])
    configured = float(dcfg["score_threshold"])
    at_configured = min(results, key=lambda r: abs(r["score_threshold"] - configured))

    header = (f"{'thresh':>7} | {'person MAE':>10} {'bias':>6} {'acc':>6} {'exact':>6} | "
              f"{'vehicle MAE':>11} {'bias':>6} {'acc':>6} {'exact':>6} | {'mean MAE':>8}")
    print(header)
    print("-" * len(header))
    for row in results:
        p, v = row["person"], row["vehicle"]
        marker = "  <- best" if row is best else ""
        print(f"{row['score_threshold']:>7.2f} | {p['mae']:>10.2f} {p['bias']:>+6.2f} "
              f"{p['accuracy']:>6.2f} {p['exact_frame_rate']:>6.2f} | "
              f"{v['mae']:>11.2f} {v['bias']:>+6.2f} {v['accuracy']:>6.2f} "
              f"{v['exact_frame_rate']:>6.2f} | {row['combined_mae']:>8.2f}{marker}")
    print()
    print(f"configured threshold {configured:.2f}: "
          f"person MAE {at_configured['person']['mae']:.2f} "
          f"(accuracy {at_configured['person']['accuracy']:.2f}), "
          f"vehicle MAE {at_configured['vehicle']['mae']:.2f} "
          f"(accuracy {at_configured['vehicle']['accuracy']:.2f})")
    print(f"best threshold on this ground truth: {best['score_threshold']:.2f} "
          f"(mean MAE {best['combined_mae']:.2f})")

    payload = {
        "source": str(source),
        "model": args.model,
        "tile": tile,
        "ground_truth_file": str(resolve(args.gt)),
        "labelled_frames": frame_ids,
        "configured_threshold": configured,
        "best_threshold": best["score_threshold"],
        "sweep": results,
        "caveats": [
            f"Only {len(frame_ids)} frames are labelled, so every number here has wide "
            "error bars; treat it as a sanity check, not a benchmark.",
            "This is count accuracy, not detection accuracy: no box matching is done, "
            "so a frame with the right count but wrong boxes scores perfectly.",
            "The best threshold is selected on the same frames it is measured on, so it "
            "is optimistic. Label a second set of frames to check it holds.",
        ],
    }
    with open(outdir / "eval_counts.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    lines = ["# Counting accuracy vs manual ground truth", "",
             f"- Source: `{source}`",
             f"- Model: `{args.model}`{' with 2x2 tiling' if tile else ''}",
             f"- Labelled frames: {len(frame_ids)} (`{args.gt}`)",
             f"- Configured threshold: {configured:.2f}",
             "",
             "`accuracy = 1 - MAE / mean(ground truth)`, clipped at 0. "
             "`bias` is mean signed error; negative means the model under-counts. "
             "`exact` is the fraction of frames counted exactly right.",
             "",
             "| threshold | person MAE | person bias | person acc | person exact | "
             "vehicle MAE | vehicle bias | vehicle acc | vehicle exact | mean MAE |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in results:
        p, v = row["person"], row["vehicle"]
        bold = "**" if row is best else ""
        lines.append(
            f"| {bold}{row['score_threshold']:.2f}{bold} | {p['mae']:.2f} | {p['bias']:+.2f} | "
            f"{p['accuracy']:.2f} | {p['exact_frame_rate']:.2f} | {v['mae']:.2f} | "
            f"{v['bias']:+.2f} | {v['accuracy']:.2f} | {v['exact_frame_rate']:.2f} | "
            f"{row['combined_mae']:.2f} |"
        )
    lines += ["", f"Best mean MAE at threshold **{best['score_threshold']:.2f}**.", "",
              "## Per-frame at the configured threshold", "",
              "| frame | person pred | person GT | vehicle pred | vehicle GT |",
              "| --- | --- | --- | --- | --- |"]
    for row in at_configured["per_frame"]:
        lines.append(f"| {row['frame_idx']} | {row['person_pred']} | {row['person_gt']} | "
                     f"{row['vehicle_pred']} | {row['vehicle_gt']} |")
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in payload["caveats"]] + [""]
    (outdir / "eval_counts.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"wrote {outdir / 'eval_counts.md'} and {outdir / 'eval_counts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
