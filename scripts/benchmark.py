"""Benchmark FP32 vs INT8: model-only latency, program size, memory, and counts.

Two layers, because they answer different questions:

* ``--micro`` (default on): a tight loop over the forward pass only, with
  warm-up iterations discarded. This is the number to quote for "how fast is
  the model", and it is the one that transfers to a Raspberry Pi.
* full pipeline runs (default on): each precision is run through
  ``scripts/run_pipeline.py``, which writes ``outputs/bench_<precision>.json``.
  Those files are what fills the FP32-vs-INT8 table in ``outputs/report.md``,
  and they also show whether quantization changed the *counts*, not just the
  speed.

    python scripts/benchmark.py                      # both layers, both precisions
    python scripts/benchmark.py --micro-only -n 50
    python scripts/benchmark.py --stride 5 --tile
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
# Python adds the script's own directory to sys.path, but not under -I/-P.
sys.path.insert(0, str(_HERE))

from safecross import load_config, load_meta, resolve  # noqa: E402
from safecross.preprocess import preprocess_frame  # noqa: E402
from safecross.runtime import build_runtime  # noqa: E402
from safecross.video import sample_frames  # noqa: E402

import run_pipeline  # noqa: E402  (same directory)


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def micro_benchmark(precision: str, cfg: dict, tensors: list[np.ndarray],
                    warmup: int, iters: int) -> dict:
    """Time the forward pass alone, cycling through real frames."""
    baseline = _rss_mb()
    runtime = build_runtime(precision, cfg)
    peak = _rss_mb()

    for i in range(warmup):
        runtime(tensors[i % len(tensors)])

    times_ms: list[float] = []
    for i in range(iters):
        tensor = tensors[i % len(tensors)]
        start = time.perf_counter()
        runtime(tensor)
        times_ms.append((time.perf_counter() - start) * 1000.0)
        peak = max(peak, _rss_mb())

    size_mb = runtime.size_mb
    model_path = str(runtime.model_path) if runtime.model_path else "n/a (eager)"
    runtime.close()

    return {
        "precision": precision,
        "model_path": model_path,
        "size_mb": size_mb,
        "warmup_iters": warmup,
        "iters": iters,
        "avg_ms": float(statistics.fmean(times_ms)),
        "p50_ms": float(run_pipeline._percentile(times_ms, 50)),
        "p95_ms": float(run_pipeline._percentile(times_ms, 95)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "fps": 1000.0 / float(statistics.fmean(times_ms)),
        "baseline_rss_mb": baseline,
        "peak_rss_mb": peak,
    }


def interleaved_benchmark(precisions: list[str], cfg: dict, tensors: list[np.ndarray],
                          warmup: int, iters: int) -> list[dict]:
    """Time every precision in one interleaved loop.

    Running one precision to completion and then the next makes the result
    depend on run order: on a laptop, sustained AVX load drops the clock, so
    whichever model runs second looks a few percent slower. Measured on this
    machine that bias was ~2.7 ms, which is the same size as the FP32/INT8
    difference being measured. Alternating the precisions inside a single loop
    spreads any thermal or frequency drift evenly across all of them.
    """
    runtimes = {}
    baseline = _rss_mb()
    for precision in precisions:
        runtimes[precision] = build_runtime(precision, cfg)
    peak = _rss_mb()

    for i in range(warmup):
        for runtime in runtimes.values():
            runtime(tensors[i % len(tensors)])

    times: dict[str, list[float]] = {p: [] for p in precisions}
    for i in range(iters):
        tensor = tensors[i % len(tensors)]
        for precision, runtime in runtimes.items():
            start = time.perf_counter()
            runtime(tensor)
            times[precision].append((time.perf_counter() - start) * 1000.0)
        peak = max(peak, _rss_mb())

    results = []
    for precision, runtime in runtimes.items():
        series = times[precision]
        results.append({
            "precision": precision,
            "model_path": str(runtime.model_path) if runtime.model_path else "n/a (eager)",
            "size_mb": runtime.size_mb,
            "warmup_iters": warmup,
            "iters": iters,
            "interleaved": True,
            "avg_ms": float(statistics.fmean(series)),
            "p50_ms": float(run_pipeline._percentile(series, 50)),
            "p95_ms": float(run_pipeline._percentile(series, 95)),
            "min_ms": float(min(series)),
            "max_ms": float(max(series)),
            "fps": 1000.0 / float(statistics.fmean(series)),
            "baseline_rss_mb": baseline,
            "peak_rss_mb": peak,
        })
        runtime.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/test_video.mp4")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--precisions", default="fp32,int8",
                        help="comma separated: fp32, int8, eager")
    parser.add_argument("-n", "--iters", type=int, default=30, help="timed forward passes")
    parser.add_argument("--warmup", type=int, default=5, help="discarded forward passes")
    parser.add_argument("--frames", type=int, default=8, help="distinct frames to cycle through")
    parser.add_argument("--stride", type=int, default=10, help="stride for the full pipeline runs")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tile", action="store_true",
                        help="tile the full pipeline runs (ignored when --matrix is set)")
    parser.add_argument("--matrix", action="store_true", default=True,
                        help="run every precision x {no-tile, tile} combination (default)")
    parser.add_argument("--no-matrix", dest="matrix", action="store_false",
                        help="run only the single --tile setting given")
    parser.add_argument("--micro-only", action="store_true",
                        help="skip the full pipeline runs")
    parser.add_argument("--sequential", action="store_true",
                        help="time each precision in its own loop instead of interleaving "
                             "them (exposes run-order/thermal bias; interleaved is default)")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    outdir = resolve(args.outdir or cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(resolve(cfg["model"]["meta"]))
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]

    source = resolve(args.source)
    frames = sample_frames(source, args.frames)
    if not frames:
        raise SystemExit(f"no frames read from {source}")
    tensors = [
        preprocess_frame(f.image, size=tuple(meta["input_size"]),
                         mean=meta["image_mean"], std=meta["image_std"])
        for f in frames
    ]

    print(f"micro-benchmark: {args.iters} timed passes after {args.warmup} warm-up, "
          f"cycling {len(tensors)} frames from {source.name}")
    print(f"host: {platform.system()} {platform.release()} ({platform.machine()}), "
          f"{__import__('os').cpu_count()} logical CPUs")
    print("-" * 88)

    micro: list[dict] = []
    available = []
    for precision in precisions:
        try:
            build_runtime(precision, cfg).close()
            available.append(precision)
        except FileNotFoundError as exc:
            print(f"{precision}: skipped -- {exc}")

    if args.sequential or len(available) < 2:
        for precision in available:
            micro.append(micro_benchmark(precision, cfg, tensors, args.warmup, args.iters))
    else:
        micro = interleaved_benchmark(available, cfg, tensors, args.warmup, args.iters)

    for result in micro:
        print(f"{result['precision']:>6}  {result['size_mb']:7.2f} MB  "
              f"avg {result['avg_ms']:7.2f} ms  p50 {result['p50_ms']:7.2f}  "
              f"p95 {result['p95_ms']:7.2f}  {result['fps']:6.2f} FPS  "
              f"peak RSS {result['peak_rss_mb']:7.1f} MB")

    if len(micro) == 2:
        ordered = sorted(micro, key=lambda r: r["avg_ms"])
        fast, slow = ordered[0], ordered[-1]
        ratio = slow["avg_ms"] / fast["avg_ms"]
        print("-" * 88)
        if ratio < 1.05:
            print(f"{fast['precision']} and {slow['precision']} are within "
                  f"{(ratio - 1) * 100:.1f}% of each other -- no measurable latency "
                  f"difference at this batch size on this CPU")
        else:
            print(f"{fast['precision']} is {ratio:.2f}x faster than {slow['precision']}")
        big = max(micro, key=lambda r: r["size_mb"])
        small = min(micro, key=lambda r: r["size_mb"])
        print(f"program file: {small['precision']} is "
              f"{big['size_mb'] / small['size_mb']:.2f}x smaller "
              f"({small['size_mb']:.2f} MB vs {big['size_mb']:.2f} MB)")

    pipeline_runs: list[str] = []
    if not args.micro_only:
        combos = (
            [(p, t) for p in available for t in (False, True)]
            if args.matrix else [(p, args.tile) for p in available]
        )
        for precision, tile in combos:
            label = f"{precision}{' +tile' if tile else ''}"
            print()
            print("=" * 88)
            print(f"full pipeline run: {label}")
            print("=" * 88)
            # --no-save-frames keeps JPEG encoding out of the end-to-end figure,
            # which otherwise dominates it and hides the model cost.
            run_argv = [
                "--source", str(source), "--model", precision,
                "--stride", str(args.stride), "--config", args.config,
                "--outdir", str(outdir), "--quiet",
                "--no-save-frames", "--no-video", "--no-detections",
            ]
            if tile:
                run_argv.append("--tile")
            if args.max_frames is not None:
                run_argv += ["--max-frames", str(args.max_frames)]
            try:
                run_pipeline.main(run_argv)
                pipeline_runs.append(label)
            except SystemExit as exc:
                print(f"{label}: pipeline run failed -- {exc}")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "python": platform.python_version(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "source": str(source),
        "micro": micro,
        "pipeline_runs": pipeline_runs,
    }
    with open(outdir / "benchmark.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = ["# SafeCross benchmark", "",
             f"*{summary['generated_at']} -- {summary['host']['platform']}, "
             f"{summary['host']['cpu_count']} logical CPUs, Python {summary['host']['python']}*",
             "",
             "## Model-only forward pass",
             "",
             f"{args.iters} timed passes after {args.warmup} warm-up passes, "
             f"cycling {len(tensors)} real frames. Single image per pass, batch size 1. "
             + ("Precisions were timed in separate loops, so run-order and thermal "
                "drift are NOT cancelled."
                if (args.sequential or len(micro) < 2) else
                "Precisions were interleaved inside one loop, so run-order and "
                "thermal drift affect all of them equally."),
             "",
             "| Precision | .pte MB | avg ms | p50 ms | p95 ms | min ms | max ms | FPS | peak RSS MB |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in micro:
        lines.append(
            f"| {r['precision']} | {r['size_mb']:.2f} | {r['avg_ms']:.2f} | {r['p50_ms']:.2f} | "
            f"{r['p95_ms']:.2f} | {r['min_ms']:.2f} | {r['max_ms']:.2f} | {r['fps']:.2f} | "
            f"{r['peak_rss_mb']:.1f} |"
        )
    lines += ["",
              "Peak RSS is the whole Python process, so it includes the interpreter, "
              "NumPy, OpenCV and torch. It is an upper bound on what the model needs, "
              "not the model's own footprint.",
              ""]
    if pipeline_runs:
        rows = run_pipeline.build_comparison(outdir)
        lines += ["## Full pipeline",
                  "",
                  f"Ran: {', '.join(pipeline_runs)}. Annotated frames and video were "
                  "disabled so that JPEG encoding does not dominate the end-to-end "
                  "figure.",
                  "",
                  "| Config | .pte MB | model avg ms | model p95 ms | model FPS | "
                  "end-to-end avg ms | end-to-end FPS | peak RSS MB | mean person | "
                  "mean vehicle_total | flagged frames |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            r, t, m, c = row["run"], row["timing"], row["memory"], row["counts"]
            label = run_pipeline.config_label(r)
            model_fps = 1000.0 / t["model_ms"]["avg"] if t["model_ms"]["avg"] else float("nan")
            e2e_fps = (1000.0 / t["end_to_end_ms"]["avg"]
                       if t["end_to_end_ms"]["avg"] else float("nan"))
            person = c["per_class"].get("person", {}).get("mean", float("nan"))
            lines.append(
                f"| {label} | {r['size_mb']:.2f} | {t['model_ms']['avg']:.1f} | "
                f"{t['model_ms']['p95']:.1f} | {model_fps:.2f} | "
                f"{t['end_to_end_ms']['avg']:.1f} | {e2e_fps:.2f} | "
                f"{m['peak_rss_mb']:.0f} | {person:.2f} | "
                f"{c['vehicle_total']['mean']:.2f} | {c['conflict_frames']} |"
            )
        lines += ["", "See `report.md` for the same table plus the conflict-flag "
                      "breakdown of the most recent run.", ""]
    (outdir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"wrote {outdir / 'benchmark.md'} and {outdir / 'benchmark.json'}")
    if pipeline_runs:
        print(f"comparison table refreshed in {outdir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
