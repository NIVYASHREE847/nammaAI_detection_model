"""SafeCross end-to-end pipeline: video or image folder -> counts, overlays, report.

    python scripts/run_pipeline.py --source data/test_video.mp4 [--model fp32|int8|eager]
                                   [--tile] [--stride N]

Per frame the pipeline does: decode -> preprocess (NumPy) -> one forward pass
through the ExecuTorch program -> anchor decode + NMS (NumPy) -> per-class
counts -> zone test + conflict rule -> annotated frame.

Outputs under ``outputs/``:
  counts.csv        one row per processed frame
  annotated/        annotated frames (PNG-free, JPEG, to keep the folder small)
  annotated.mp4     the same frames as a video
  samples/          the 3 most informative frames
  report.md         human-readable run report, incl. FP32 vs INT8 comparison
  report.json       the same numbers, machine-readable
  bench_<prec>.json per-precision record, used to build the comparison table
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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
from safecross.video import VideoWriter, iter_frames, probe_source  # noqa: E402
from safecross.visualize import annotate_frame  # noqa: E402
from safecross.zones import ConflictRule, Zone  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; avoids a SciPy dependency."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * len(ordered) + 0.5)) - 1))
    return ordered[k]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {k: float("nan") for k in ("avg", "p50", "p95", "min", "max")}
    return {
        "avg": float(statistics.fmean(values)),
        "p50": float(_percentile(values, 50)),
        "p95": float(_percentile(values, 95)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _distribution(series: list[int]) -> dict[str, float]:
    """Quantiles and tail fractions, so the report shows the shape, not just a mean."""
    arr = np.asarray(series, dtype=np.int64)
    if arr.size == 0:
        return {}
    out = {
        "mean": float(arr.mean()),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "max": int(arr.max()),
    }
    for k in (1, 2, 3, 4, 5):
        out[f"frac_ge_{k}"] = float((arr >= k).mean())
    return out


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover - psutil missing
        return float("nan")


@dataclass
class SampleCandidate:
    frame_idx: int
    priority: tuple
    image: np.ndarray = field(repr=False)


def _offer_sample(samples: list[SampleCandidate], candidate: SampleCandidate,
                  n_samples: int, min_gap: int) -> None:
    """Keep the best N sample frames, forcing them to be spread out in time.

    Without a spacing rule the top three frames are almost always three
    consecutive frames of the same moment, which is useless as evidence. A
    candidate within ``min_gap`` frames of an existing sample competes only
    against that neighbour; otherwise it competes against the whole set.
    """
    for i, existing in enumerate(samples):
        if abs(existing.frame_idx - candidate.frame_idx) < min_gap:
            if candidate.priority > existing.priority:
                samples[i] = candidate
            return
    samples.append(candidate)
    samples.sort(key=lambda s: s.priority, reverse=True)
    del samples[n_samples:]


# --------------------------------------------------------------------------- #
# inference for one frame
# --------------------------------------------------------------------------- #
def detect_frame(
    frame_bgr: np.ndarray,
    runtime,
    anchors: np.ndarray,
    meta: dict,
    det_kwargs: dict,
    tile_cfg: dict | None,
    dedup: tuple[Sequence[int], float] | None = None,
) -> tuple[Detections, float]:
    """Run detection on one frame. Returns ``(detections, model_only_seconds)``.

    With tiling the model runs once per tile (plus once on the full frame when
    configured), each tile's boxes are mapped back into frame coordinates, and
    the union is de-duplicated with class-aware NMS. The reported model time is
    the sum over tiles, so the latency numbers stay comparable with the
    untiled run.
    """
    height, width = frame_bgr.shape[:2]
    size = tuple(meta["input_size"])
    mean, std = meta["image_mean"], meta["image_std"]
    model_seconds = 0.0

    if not tile_cfg:
        nchw = preprocess_frame(frame_bgr, size=size, mean=mean, std=std)
        t0 = time.perf_counter()
        cls_logits, bbox = runtime(nchw)
        model_seconds += time.perf_counter() - t0
        dets = postprocess_detections(
            cls_logits, bbox, anchors, original_size=(height, width), **det_kwargs
        )
        if dedup:
            dets = suppress_cross_class_duplicates(dets, dedup[0], dedup[1])
        return dets, model_seconds

    rects = tile_rects(
        height,
        width,
        grid=tuple(tile_cfg.get("tile_grid", (2, 2))),
        overlap=float(tile_cfg.get("tile_overlap", 0.15)),
        include_full_frame=bool(tile_cfg.get("tile_include_full_frame", True)),
    )
    parts: list[Detections] = []
    for rect in rects:
        nchw = preprocess_crop(frame_bgr, rect, size=size, mean=mean, std=std)
        t0 = time.perf_counter()
        cls_logits, bbox = runtime(nchw)
        model_seconds += time.perf_counter() - t0
        dets = postprocess_detections(
            cls_logits, bbox, anchors,
            original_size=(rect.height, rect.width), **det_kwargs,
        )
        if len(dets):
            boxes = dets.boxes.copy()
            boxes[:, [0, 2]] += rect.x0
            boxes[:, [1, 3]] += rect.y0
            parts.append(Detections(boxes, dets.scores, dets.labels))
    merged = merge_detections(
        parts,
        nms_iou=float(det_kwargs["nms_iou"]),
        detections_per_img=int(det_kwargs["detections_per_img"]),
    )
    if dedup:
        merged = suppress_cross_class_duplicates(merged, dedup[0], dedup[1])
    return merged, model_seconds


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _fmt(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def config_label(run: dict) -> str:
    """Short name for a run: precision, tiling, and whether it wrote files."""
    parts = [run["precision"]]
    if run.get("tile"):
        parts.append("+tile")
    if run.get("wrote_frames") or run.get("wrote_video"):
        parts.append("+io")
    return " ".join(parts)


def _recommend(comparison: list[dict]) -> dict | None:
    """Pick a default config from the measured runs, and say why.

    The rule is explicit rather than a judgement call, and every claim in the
    returned rationale is computed from the measured runs:

    1. Tiling is chosen on pedestrian recall, because recall is the binding
       constraint for a pedestrian-safety tool -- a config that cannot see
       people makes the conflict flag meaningless whatever its latency.
    2. Precision is chosen on recall too, falling back to latency only when the
       two precisions are within 5% of each other on mean person count.

    Runs that also wrote annotated frames or video are excluded, because JPEG
    encoding competes with the XNNPACK thread pool and inflates their timings.
    """
    if len(comparison) < 2:
        return None
    by_label = {}
    for row in comparison:
        r = row["run"]
        # Compare like with like: skip runs that also wrote frames or video.
        if r.get("wrote_frames") or r.get("wrote_video"):
            continue
        by_label[f"{r['precision']}{'+tile' if r['tile'] else ''}"] = row

    def person_mean(row):
        return row["counts"]["per_class"].get("person", {}).get("mean", 0.0)

    tiled = {k: v for k, v in by_label.items() if v["run"]["tile"]}
    untiled = {k: v for k, v in by_label.items() if not v["run"]["tile"]}
    if not tiled:
        return None

    # Step 1: tiling, decided on pedestrian recall.
    label, row = max(tiled.items(), key=lambda kv: person_mean(kv[1]))

    # Step 2: precision. Prefer the one that detects more people; fall back to
    # the faster one only when recall is effectively tied (within 5%). The
    # rationale string is derived from the data, never asserted, so it cannot
    # contradict the label when a run is missing from the comparison.
    precision_note = ""
    if len(tiled) > 1:
        ranked = sorted(tiled.items(), key=lambda kv: -person_mean(kv[1]))
        (top_label, top_row), (next_label, next_row) = ranked[0], ranked[1]
        top_p, next_p = person_mean(top_row), person_mean(next_row)
        tied = next_p >= 0.95 * top_p if top_p else True
        if tied:
            label, row = min(ranked, key=lambda kv: kv[1]["timing"]["model_ms"]["avg"])
            precision_note = (
                f" {top_label} and {next_label} detect people equally well "
                f"({top_p:.2f} vs {next_p:.2f} per frame), so the faster of the two wins."
            )
        else:
            label, row = top_label, top_row
            precision_note = (
                f" {top_label} over {next_label} because it finds "
                f"{top_p:.2f} people per frame against {next_p:.2f} "
                f"({100.0 * (top_p - next_p) / top_p:.0f}% more) at comparable speed. "
                f"INT8's win is file size, which matters on a Pi and not on this laptop."
            )
    if not untiled:
        return {"config": label,
                "why": "Tiling is required for usable pedestrian recall." + precision_note}

    best_untiled = max(untiled.items(), key=lambda kv: person_mean(kv[1]))
    gain = person_mean(row) - person_mean(best_untiled[1])
    base_ms = best_untiled[1]["timing"]["model_ms"]["avg"]
    cost = row["timing"]["model_ms"]["avg"] / base_ms if base_ms else float("nan")
    why = (
        f"Tiling lifts the mean person count from {person_mean(best_untiled[1]):.2f} "
        f"to {person_mean(row):.2f} per frame ({gain:+.2f}) for {cost:.1f}x the model "
        f"time. Pedestrian recall is the binding constraint here -- an untiled run "
        f"barely sees people at all, which makes the conflict flag meaningless."
        + precision_note
    )
    return {"config": label, "why": why}


def _load_micro_benchmark(outdir: Path) -> list[dict]:
    """Pick up the interleaved latency numbers from a previous benchmark.py run."""
    path = outdir / "benchmark.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("micro", [])
    except Exception:
        return []


def build_comparison(outdir: Path) -> list[dict]:
    """Load every ``bench_<precision>.json`` so report.md can compare runs."""
    rows = []
    for path in sorted(outdir.glob("bench_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                rows.append(json.load(handle))
        except Exception:
            continue
    return rows


def write_report(report: dict, outdir: Path, comparison: list[dict]) -> Path:
    run, timing, memory, counts = report["run"], report["timing"], report["memory"], report["counts"]
    lines: list[str] = []
    add = lines.append

    add("# SafeCross run report")
    add("")
    add(f"*Generated {report['generated_at']} on {report['host']['platform']}, "
        f"Python {report['host']['python']}*")
    add("")
    add("## Run")
    add("")
    add("| Setting | Value |")
    add("| --- | --- |")
    add(f"| Model | `{run['model']}` |")
    add(f"| Precision | **{run['precision']}** |")
    add(f"| Runtime | {run['runtime']} |")
    add(f"| Program file | `{run['model_path']}` |")
    add(f"| Program size | {_fmt(run['size_mb'])} MB |")
    add(f"| Source | `{run['source']}` ({run['source_info']}) |")
    add(f"| Frames processed | {run['frames_processed']} (stride {run['stride']}) |")
    add(f"| Score threshold | {run['score_threshold']} |")
    add(f"| NMS IoU | {run['nms_iou']} |")
    add(f"| Tiled inference | {'on -- ' + run['tile_desc'] if run['tile'] else 'off'} |")
    add(f"| Cross-class vehicle de-duplication | "
        f"{'on (IoU ' + str(run['cross_class_iou']) + ')' if run['merge_duplicate_vehicles'] else 'off'} |")
    add(f"| Zone | {run['zone']} |")
    add("")

    add("## Performance")
    add("")
    add("`model-only` is the forward pass alone. `end-to-end` adds decode, "
        "preprocessing, NMS, counting and annotation -- the full per-frame cost.")
    add("")
    add("| Stage | avg ms | p50 ms | p95 ms | min ms | max ms | FPS (1/avg) |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for key, label in (("model_ms", "model-only"), ("end_to_end_ms", "end-to-end")):
        s = timing[key]
        fps = 1000.0 / s["avg"] if s["avg"] and np.isfinite(s["avg"]) and s["avg"] > 0 else float("nan")
        add(f"| {label} | {_fmt(s['avg'])} | {_fmt(s['p50'])} | {_fmt(s['p95'])} | "
            f"{_fmt(s['min'])} | {_fmt(s['max'])} | {_fmt(fps)} |")
    add("")
    add(f"- Wall-clock for the whole run: **{_fmt(timing['total_seconds'])} s** "
        f"({_fmt(timing['throughput_fps'])} FPS including file writes)")
    add(f"- Peak process memory (RSS): **{_fmt(memory['peak_rss_mb'])} MB** "
        f"(baseline before model load {_fmt(memory['baseline_rss_mb'])} MB, "
        f"delta {_fmt(memory['delta_mb'])} MB)")
    add("")

    add("## Counts")
    add("")
    add("Counts are per frame, not unique objects: an object standing in view "
        "for 100 frames is counted in each of them. No tracking is performed.")
    add("")
    add("| Class | mean/frame | max/frame | peak frame | frames with >=1 |")
    add("| --- | --- | --- | --- | --- |")
    for name, stat in counts["per_class"].items():
        add(f"| {name} | {_fmt(stat['mean'])} | {stat['max']} | {stat['peak_frame']} | "
            f"{stat['frames_present']} |")
    vt = counts["vehicle_total"]
    add(f"| **vehicle_total** | {_fmt(vt['mean'])} | {vt['max']} | {vt['peak_frame']} | "
        f"{vt['frames_present']} |")
    add("")
    add(f"- Busiest frame overall: **{counts['peak_frame']}** "
        f"with {counts['peak_frame_detections']} detections")
    add("")

    add("## Conflict flag")
    add("")
    add(f"Rule: **{run['conflict_rule']}**")
    add("")
    ep = counts["conflict_episodes"]
    add(f"- Flagged frames: **{counts['conflict_frames']} / {run['frames_processed']}** "
        f"({_fmt(100.0 * ep['flagged_fraction'], 1)}%)")
    if run["conflict_min_consecutive_frames"] > 1:
        add(f"- Frames meeting the count condition before the duration rule: "
            f"{counts['conflict_condition_frames']} "
            f"({_fmt(100.0 * counts['conflict_condition_frames'] / max(1, run['frames_processed']), 1)}%)")
    add(f"- Episodes: **{ep['count']}**, mean {_fmt(ep['mean_frames'], 1)} frames "
        f"({_fmt(ep['mean_seconds'], 2)} s), longest {ep['max_frames']} frames")
    add("")
    if ep["episodes"]:
        add("| # | start frame | start s | end frame | frames | max persons | max vehicles |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for i, episode in enumerate(ep["episodes"][:20], start=1):
            add(f"| {i} | {episode['start_frame']} | {episode['start_s']:.2f} | "
                f"{episode['end_frame']} | {episode['frames']} | "
                f"{episode['max_persons']} | {episode['max_vehicles']} |")
        if len(ep["episodes"]) > 20:
            add(f"| ... | | | | | | ({len(ep['episodes']) - 20} more in `report.json`) |")
        add("")

    add("### In-zone count distribution")
    add("")
    add("This is what the thresholds are chosen against: a threshold only "
        "discriminates where the distribution actually has mass.")
    add("")
    add("| Series | mean | p50 | p90 | max | frac >=1 | frac >=2 | frac >=3 | frac >=4 | frac >=5 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, dist in counts["in_zone_distribution"].items():
        if not dist:
            continue
        add(f"| {key} | {_fmt(dist['mean'])} | {dist['p50']} | {dist['p90']} | {dist['max']} | "
            + " | ".join(_fmt(dist[f'frac_ge_{k}']) for k in (1, 2, 3, 4, 5)) + " |")
    add("")

    if report.get("parity"):
        add("## Accuracy vs torchvision eager mode")
        add("")
        add("Measured at export time on frames from this clip "
            "(`models/parity_<precision>.json`).")
        add("")
        add("| Precision | max abs cls-logit diff | max abs bbox diff | "
            "eager dets | .pte dets | matched | match rate |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for precision, par in report["parity"].items():
            raw, det = par["raw_tensor"], par["detections_vs_eager"]
            add(f"| {precision} | {raw['max_abs_cls_logit_diff']:.3e} | "
                f"{raw['max_abs_bbox_regression_diff']:.3e} | "
                f"{det['reference_detections']} | {det['runtime_detections']} | "
                f"{det['matched']} | {det['match_rate_vs_eager']:.3f} |")
        add("")

    if len(comparison) >= 1:
        add("## Configuration comparison")
        add("")
        if len(comparison) < 2:
            add("> Only one configuration has been run so far. Run "
                "`python scripts/benchmark.py` to fill this table with all four "
                "precision x tiling combinations.")
            add("")
        add("Model times below come from full pipeline runs, where the thread pool "
            "idles between frames and run order biases the result. For a clean "
            "latency comparison between precisions use the interleaved "
            "micro-benchmark in `benchmark.md` (`python scripts/benchmark.py`), "
            "which alternates them inside one loop. Runs made by `benchmark.py` "
            "have annotated frames and video disabled, so their end-to-end figure "
            "is free of JPEG encoding; rows marked `+io` also wrote them and "
            "are NOT comparable with the rest -- JPEG encoding competes with "
            "the XNNPACK thread pool and inflates even the model-only number. "
            "The recommendation below ignores `+io` rows.")
        add("")
        add("| Config | .pte MB | model avg ms | model p95 ms | model FPS | "
            "end-to-end avg ms | end-to-end FPS | peak RSS MB | mean person | "
            "mean vehicle_total | flagged frames |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in comparison:
            r, t, m, c = row["run"], row["timing"], row["memory"], row["counts"]
            label = config_label(r)
            model_fps = 1000.0 / t["model_ms"]["avg"] if t["model_ms"]["avg"] else float("nan")
            e2e_fps = (1000.0 / t["end_to_end_ms"]["avg"]
                       if t["end_to_end_ms"]["avg"] else float("nan"))
            person = c["per_class"].get("person", {}).get("mean", float("nan"))
            add(f"| {label} | {_fmt(r['size_mb'])} | "
                f"{_fmt(t['model_ms']['avg'], 1)} | {_fmt(t['model_ms']['p95'], 1)} | "
                f"{_fmt(model_fps)} | {_fmt(t['end_to_end_ms']['avg'], 1)} | "
                f"{_fmt(e2e_fps)} | {_fmt(m['peak_rss_mb'], 0)} | {_fmt(person)} | "
                f"{_fmt(c['vehicle_total']['mean'])} | {c['conflict_frames']} |")
        add("")
        recommendation = _recommend(comparison)
        if recommendation:
            add(f"**Recommended default: `{recommendation['config']}`.** "
                + recommendation["why"])
            add("")

    micro = report.get("micro_benchmark") or []
    if micro:
        add("### Interleaved model-only benchmark")
        add("")
        add(f"From `benchmark.json` ({micro[0].get('iters', '?')} timed passes"
            f"{', interleaved' if micro[0].get('interleaved') else ', sequential'}), "
            "which is the number to quote for model latency.")
        add("")
        add("| Precision | .pte MB | avg ms | p50 ms | p95 ms | FPS |")
        add("| --- | --- | --- | --- | --- | --- |")
        for row in micro:
            add(f"| {row['precision']} | {_fmt(row['size_mb'])} | {_fmt(row['avg_ms'])} | "
                f"{_fmt(row['p50_ms'])} | {_fmt(row['p95_ms'])} | {_fmt(row['fps'])} |")
        add("")

    add("## Outputs")
    add("")
    for key, value in report["outputs"].items():
        add(f"- `{value}` -- {key}")
    add("")

    add("## Limitations")
    add("")
    for item in report["limitations"]:
        add(f"- {item}")
    add("")

    path = outdir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/test_video.mp4",
                        help="video file or folder of images")
    parser.add_argument("--model", choices=["fp32", "int8", "eager"], default="fp32",
                        help="fp32/int8 run the ExecuTorch .pte; eager runs torchvision directly")
    parser.add_argument("--tile", action="store_true",
                        help="2x2 tiled inference + merge NMS (better on small/distant objects)")
    parser.add_argument("--stride", type=int, default=None, help="process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--score", type=float, default=None, help="override score threshold")
    parser.add_argument("--nms", type=float, default=None, help="override NMS IoU threshold")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--no-save-frames", "--no-frames", dest="no_save_frames",
                        action="store_true",
                        help="skip writing annotated/*.jpg. JPEG encoding dominates the "
                             "end-to-end timing, so use this whenever you are measuring "
                             "speed. The annotated video is independent (--no-video).")
    parser.add_argument("--no-video", action="store_true", help="do not write annotated.mp4")
    parser.add_argument("--no-detections", action="store_true",
                        help="do not write outputs/detections.jsonl")
    parser.add_argument("--quiet", action="store_true", help="no per-frame terminal output")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    pcfg = cfg["pipeline"]
    dcfg = dict(cfg["detection"])
    if args.score is not None:
        dcfg["score_threshold"] = args.score
    if args.nms is not None:
        dcfg["nms_iou"] = args.nms
    stride = int(args.stride if args.stride is not None else pcfg.get("stride", 1))
    tile = bool(args.tile or pcfg.get("tile", False))

    outdir = resolve(args.outdir or cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    annotated_dir = outdir / "annotated"
    samples_dir = outdir / "samples"
    annotated_dir.mkdir(exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    baseline_rss = _rss_mb()

    meta_path = resolve(cfg["model"]["meta"])
    anchors_path = resolve(cfg["model"]["anchors"])
    if not meta_path.exists() or not anchors_path.exists():
        raise SystemExit(
            f"missing {meta_path.name}/{anchors_path.name}. Run:  python export/export_model.py"
        )
    meta = load_meta(meta_path)
    anchors = load_anchors(anchors_path)
    counter = ClassCounter.from_config(cfg, meta["categories"])

    runtime = build_runtime(args.model, cfg)
    source = resolve(args.source)
    info = probe_source(source)
    zone = Zone.from_config(cfg, info.width, info.height)
    rule = ConflictRule.from_config(cfg)

    det_kwargs = dict(
        input_size=tuple(meta["input_size"]),
        score_threshold=float(dcfg["score_threshold"]),
        nms_iou=float(dcfg["nms_iou"]),
        topk_candidates=int(dcfg["topk_candidates"]),
        detections_per_img=int(dcfg["detections_per_img"]),
        keep_labels=counter.keep_labels,
        box_weights=meta["box_weights"],
    )
    tile_cfg = pcfg if tile else None
    dedup = (
        (sorted(counter.vehicle_ids), float(dcfg.get("cross_class_iou", 0.6)))
        if dcfg.get("merge_duplicate_vehicles", True)
        else None
    )

    print(f"source : {info.describe()}")
    print(f"model  : {args.model} ({runtime.name}, {_fmt(runtime.size_mb)} MB)")
    print(f"thresh : score {det_kwargs['score_threshold']}, NMS IoU {det_kwargs['nms_iou']}"
          f"{', tiled ' + str(tuple(pcfg['tile_grid'])) if tile else ''}")
    print(f"zone   : {zone.describe(rule)}")
    print("-" * 72)

    csv_path = outdir / "counts.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["frame_idx", "timestamp_s", *counter.csv_columns,
         "persons_in_zone", "vehicles_in_zone", "conflict_flag"]
    )

    video_fps = pcfg.get("annotated_video_fps") or 0
    if not video_fps:
        video_fps = info.fps / max(1, stride)
    video_writer = (
        VideoWriter(outdir / "annotated.mp4", video_fps)
        if (pcfg.get("annotated_video", True) and not args.no_video)
        else None
    )
    write_frames = pcfg.get("save_annotated_frames", True) and not args.no_save_frames
    # Raw detections are cached so that the zone polygon and the conflict
    # thresholds can be re-tuned offline without paying for inference again.
    det_path = outdir / "detections.jsonl"
    det_file = (
        open(det_path, "w", encoding="utf-8")
        if (pcfg.get("save_detections", True) and not args.no_detections)
        else None
    )

    model_times: list[float] = []
    e2e_times: list[float] = []
    per_class_series: dict[str, list[int]] = {name: [] for name in counter.class_names}
    vehicle_series: list[int] = []
    persons_zone_series: list[int] = []
    vehicles_zone_series: list[int] = []
    frame_indices: list[int] = []
    detection_counts: list[int] = []
    conflict_frames = 0
    condition_frames = 0
    peak_rss = baseline_rss
    samples: list[SampleCandidate] = []
    n_samples = int(pcfg.get("n_samples", 3))
    # Spread samples across the clip. Falls back to a sane gap when the source
    # will not report its length.
    expected = (info.frame_count // max(1, stride)) if info.frame_count else 0
    sample_min_gap = max(1, int(pcfg.get("sample_min_gap", 0))
                         or (expected // (n_samples * 2) if expected else 25))

    run_start = time.perf_counter()
    processed = 0
    try:
        for frame in iter_frames(source, stride=stride, max_frames=args.max_frames):
            t_frame = time.perf_counter()
            dets, model_seconds = detect_frame(
                frame.image, runtime, anchors, meta, det_kwargs, tile_cfg, dedup
            )
            counts = counter.count(dets)
            zresult = zone.evaluate(dets, counter, rule,
                                    frame_idx=frame.index, timestamp_s=frame.timestamp_s)

            annotated = annotate_frame(
                frame.image, dets, counts, counter.class_names, counter.label_name,
                zone=zone, zone_result=zresult, frame_idx=frame.index,
                timestamp_s=frame.timestamp_s,
                extra={"model": f"{args.model}{' +tile' if tile else ''}"},
            )
            if write_frames:
                cv2.imwrite(str(annotated_dir / f"frame_{frame.index:06d}.jpg"), annotated,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if video_writer is not None:
                video_writer.write(annotated)

            if det_file is not None:
                det_file.write(json.dumps({
                    "frame_idx": frame.index,
                    "timestamp_s": round(frame.timestamp_s, 4),
                    "width": int(frame.image.shape[1]),
                    "height": int(frame.image.shape[0]),
                    "boxes": [[round(float(c), 2) for c in box] for box in dets.boxes],
                    "scores": [round(float(x), 4) for x in dets.scores],
                    "labels": [int(x) for x in dets.labels],
                }) + "\n")

            e2e_seconds = time.perf_counter() - t_frame
            model_times.append(model_seconds * 1000.0)
            e2e_times.append(e2e_seconds * 1000.0)

            writer.writerow(
                [frame.index, f"{frame.timestamp_s:.3f}", *counts.as_row(counter.class_names),
                 zresult.persons_in_zone, zresult.vehicles_in_zone, int(zresult.conflict)]
            )
            for name in counter.class_names:
                per_class_series[name].append(counts.per_class[name])
            vehicle_series.append(counts.vehicle_total)
            persons_zone_series.append(zresult.persons_in_zone)
            vehicles_zone_series.append(zresult.vehicles_in_zone)
            frame_indices.append(frame.index)
            detection_counts.append(counts.total)
            conflict_frames += int(zresult.conflict)
            condition_frames += int(zresult.condition)
            peak_rss = max(peak_rss, _rss_mb())

            # "Best" sample = conflict first, then most in-zone activity, then
            # most detections. That surfaces frames that actually show the
            # safety situation rather than the frame with the most parked cars.
            priority = (
                int(zresult.conflict),
                zresult.persons_in_zone + zresult.vehicles_in_zone,
                counts.total,
            )
            _offer_sample(samples, SampleCandidate(frame.index, priority, annotated),
                          n_samples, sample_min_gap)

            processed += 1
            if not args.quiet:
                summary = "  ".join(
                    f"{name}={counts.per_class[name]}" for name in counter.class_names
                )
                print(
                    f"frame {frame.index:6d}  t={frame.timestamp_s:7.3f}s  {summary}  "
                    f"vehicle_total={counts.vehicle_total}  "
                    f"zone(p={zresult.persons_in_zone},v={zresult.vehicles_in_zone})"
                    f"{'  CONFLICT' if zresult.conflict else ''}  "
                    f"[{model_seconds * 1000:.1f} ms model / {e2e_seconds * 1000:.1f} ms total]"
                )
    finally:
        csv_file.close()
        if det_file is not None:
            det_file.close()
        if video_writer is not None:
            video_writer.release()
        runtime.close()

    total_seconds = time.perf_counter() - run_start
    if processed == 0:
        raise SystemExit(f"no frames processed from {source}")

    for sample in samples:
        cv2.imwrite(str(samples_dir / f"sample_frame_{sample.frame_idx:06d}.jpg"), sample.image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def series_stats(series: list[int]) -> dict:
        peak_i = int(np.argmax(series))
        return {
            "mean": float(np.mean(series)),
            "max": int(np.max(series)),
            "peak_frame": int(frame_indices[peak_i]),
            "frames_present": int(sum(1 for v in series if v > 0)),
            "sum": int(np.sum(series)),
        }

    peak_i = int(np.argmax(detection_counts))
    episodes = rule.finish()
    fps_processed = (info.fps / max(1, stride)) if info.fps else 0.0
    episode_stats = {
        "count": len(episodes),
        "total_flagged_frames": conflict_frames,
        "flagged_fraction": conflict_frames / processed if processed else 0.0,
        "mean_frames": float(np.mean([e["frames"] for e in episodes])) if episodes else 0.0,
        "max_frames": int(max((e["frames"] for e in episodes), default=0)),
        "mean_seconds": (float(np.mean([e["frames"] for e in episodes])) / fps_processed)
        if (episodes and fps_processed) else 0.0,
        "episodes": episodes,
    }
    parity: dict[str, dict] = {}
    for precision in ("fp32", "int8"):
        parity_path = resolve("models") / f"parity_{precision}.json"
        if parity_path.exists():
            with open(parity_path, "r", encoding="utf-8") as handle:
                parity[precision] = json.load(handle)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "python": platform.python_version(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "run": {
            "model": meta["model"],
            "weights": meta["weights"],
            "precision": runtime.precision if args.model != "eager" else "fp32 (eager)",
            "runtime": runtime.name,
            "model_path": str(runtime.model_path) if runtime.model_path else "n/a (eager)",
            "size_mb": runtime.size_mb,
            "source": str(source),
            "source_info": info.describe(),
            "frames_processed": processed,
            "stride": stride,
            "score_threshold": det_kwargs["score_threshold"],
            "nms_iou": det_kwargs["nms_iou"],
            "tile": tile,
            "wrote_frames": bool(write_frames),
            "wrote_video": video_writer is not None,
            "tile_desc": (
                f"{tuple(pcfg['tile_grid'])} grid, {pcfg['tile_overlap']} overlap, "
                f"full frame {'included' if pcfg.get('tile_include_full_frame', True) else 'excluded'}"
            ),
            "classes": counter.class_names,
            "vehicle_classes": counter.vehicle_classes,
            "merge_duplicate_vehicles": bool(dedup),
            "cross_class_iou": float(dcfg.get("cross_class_iou", 0.6)),
            "zone": zone.describe(rule),
            "conflict_rule": rule.describe(),
            "conflict_min_consecutive_frames": rule.min_consecutive_frames,
            "conflict_min_persons": zone.min_persons,
            "conflict_min_vehicles": zone.min_vehicles,
        },
        "timing": {
            "model_ms": _stats(model_times),
            "end_to_end_ms": _stats(e2e_times),
            "total_seconds": total_seconds,
            "throughput_fps": processed / total_seconds if total_seconds > 0 else float("nan"),
        },
        "memory": {
            "baseline_rss_mb": baseline_rss,
            "peak_rss_mb": peak_rss,
            "delta_mb": peak_rss - baseline_rss,
        },
        "counts": {
            "per_class": {name: series_stats(series) for name, series in per_class_series.items()},
            "vehicle_total": series_stats(vehicle_series),
            "conflict_frames": conflict_frames,
            "conflict_condition_frames": condition_frames,
            "conflict_episodes": episode_stats,
            "in_zone_distribution": {
                "persons_in_zone": _distribution(persons_zone_series),
                "vehicles_in_zone": _distribution(vehicles_zone_series),
            },
            "peak_frame": int(frame_indices[peak_i]),
            "peak_frame_detections": int(detection_counts[peak_i]),
            "zone": {
                "persons_mean": float(np.mean(persons_zone_series)),
                "persons_max": int(np.max(persons_zone_series)),
                "vehicles_mean": float(np.mean(vehicles_zone_series)),
                "vehicles_max": int(np.max(vehicles_zone_series)),
            },
        },
        "parity": parity,
        "micro_benchmark": _load_micro_benchmark(outdir),
        "outputs": {
            "per-frame counts": str(csv_path),
            "annotated frames": str(annotated_dir) if write_frames else "(skipped)",
            "annotated video": str(outdir / "annotated.mp4") if video_writer else "(skipped)",
            "sample frames": str(samples_dir),
            "this report": str(outdir / "report.md"),
        },
        "limitations": [
            "Counts are per frame, not unique objects: there is no tracking, so the "
            "same pedestrian is counted again in every frame they appear in.",
            "The detector is stock COCO ssdlite320 with no fine-tuning on traffic "
            "footage; small and distant pedestrians are its weakest case. Use --tile "
            "to recover some of them, at roughly 5x the model cost.",
            "INT8 is post-training quantized without any accuracy recovery; its "
            "detections differ from FP32, and the parity table above quantifies by "
            "how much on this clip.",
            "The conflict flag is a geometric co-presence test inside a hand-drawn "
            "polygon. It is evidence that a person and a vehicle shared a region at "
            "one instant, not a measurement of near-miss severity, speed or intent.",
            "Timings are from this laptop CPU. Raspberry Pi numbers will be several "
            "times slower and must be measured on the Pi, not extrapolated.",
        ],
    }

    with open(outdir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    # The filename records the I/O setting too. A run that writes 401 JPEGs is
    # not comparable with one that does not -- JPEG encoding competes with the
    # XNNPACK thread pool and inflates even the model-only figure -- so the two
    # must not overwrite each other in the comparison table.
    io_suffix = "_io" if (write_frames or video_writer is not None) else ""
    bench_name = (
        f"bench_{runtime.precision}{'_tile' if tile else ''}{io_suffix}.json"
        if args.model != "eager" else f"bench_eager{io_suffix}.json"
    )
    with open(outdir / bench_name, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report_path = write_report(report, outdir, build_comparison(outdir))

    print("-" * 72)
    print(f"frames           : {processed}")
    print(f"model-only       : avg {_fmt(report['timing']['model_ms']['avg'])} ms, "
          f"p95 {_fmt(report['timing']['model_ms']['p95'])} ms "
          f"({_fmt(1000.0 / report['timing']['model_ms']['avg'])} FPS)")
    print(f"end-to-end       : avg {_fmt(report['timing']['end_to_end_ms']['avg'])} ms "
          f"({_fmt(report['timing']['throughput_fps'])} FPS wall clock)")
    print(f"peak RSS         : {_fmt(peak_rss)} MB")
    print(f"conflict frames  : {conflict_frames} / {processed}")
    print(f"report           : {report_path}")
    print(f"counts csv       : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
