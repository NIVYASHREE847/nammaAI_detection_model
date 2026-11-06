# SafeCross

**Edge-based object detection and counting for pedestrian-safety evidence.**
Namma AI Hackathon — Problem Statement 2. Proof of concept.

SafeCross watches a fixed camera over a crossing, a school gate or a bus-stop
frontage and answers one question per frame: *how many people and how many
vehicles are in the conflict area right now?* It writes that out as a CSV, an
annotated video and a report, so a road-safety officer has dated, reviewable
evidence instead of an anecdote.

Everything runs on CPU through the **ExecuTorch** runtime with the **XNNPACK**
backend. There is no GPU anywhere in the inference path, no training, and no
cloud call. The same `.pte` program file that runs on the laptop is what you
would copy to a Raspberry Pi.

---

## What it does

| | |
| --- | --- |
| **Model** | torchvision `ssdlite320_mobilenet_v3_large`, stock COCO weights, **no training or fine-tuning** |
| **Classes** | `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, plus a `vehicle_total` roll-up |
| **Runtime** | ExecuTorch `.pte`, XNNPACK CPU backend, FP32 and INT8 |
| **Decoding** | anchor decode, score threshold and NMS done in NumPy — nothing data-dependent is inside the exported graph |
| **Zone counting** | polygon ROI from `config.yaml`, in normalized coordinates |
| **Conflict flag** | `persons_in_zone >= P AND vehicles_in_zone >= V`, both configurable |
| **`--tile`** | 2×2 overlapping tiles + merge NMS, for small/distant pedestrians |
| **Min-duration rule** | the count condition must hold for N consecutive processed frames before the flag is raised |

### Use case

A 320×320 detector on a Pi cannot police anything, and SafeCross does not try
to. What it produces is *evidence*: a time-stamped record of how often
pedestrians and moving vehicles occupy the same few square metres at a given
school gate or bus stop. That record is what turns "drivers go too fast past
the gate" into a number a municipality can act on — a count of conflict frames
per hour, with annotated stills attached, at the cost of one Raspberry Pi and
no video ever leaving the device.

---

## Setup

Python 3.12 on Windows, Linux or macOS. 64-bit ARM (Raspberry Pi OS) also has
wheels for every dependency.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# torch must come from the CPU index, or pip pulls the CUDA build
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Why these versions.** `torchvision 0.29.0` hard-pins `torch==2.14.0` (its C
extension is ABI-locked to that build). `executorch 1.5.1` works against torch
2.14 and ships `win_amd64`, `manylinux x86_64` and `manylinux aarch64` wheels
for cp312. The newer `executorch 1.6.0` forces `torch==2.15.0`, for which no
torchvision wheel exists yet — that combination breaks `import torchvision`.

> **Windows note.** The ExecuTorch wheel ships `flatc.exe`, but its internal
> lookup searches for a file named `flatc`, so the XNNPACK serializer dies with
> `FileNotFoundError: [WinError 2]`. `export/export_model.py` sets
> `FLATC_EXECUTABLE` (ExecuTorch's documented override) to the installed
> binary automatically. No action needed; no effect on Linux or macOS.

---

## One-command run

```bash
python export/export_model.py                              # once: build the .pte files
python scripts/run_pipeline.py --source data/test_video.mp4
```

The export step downloads the COCO weights, writes both `.pte` files, and
verifies them against torchvision before you trust them. Then:

```bash
# tiled inference — finds the distant pedestrians, ~5x the model cost
python scripts/run_pipeline.py --source data/test_video.mp4 --tile

# quantized model
python scripts/run_pipeline.py --source data/test_video.mp4 --model int8

# every 5th frame, folder of images instead of a video
python scripts/run_pipeline.py --source data/frames/ --stride 5

# measuring speed: skip the per-frame JPEGs, which otherwise dominate
# the end-to-end figure
python scripts/run_pipeline.py --source data/test_video.mp4 --no-save-frames --no-video

# every precision x {no-tile, tile}: latency, size, memory, counts
python scripts/benchmark.py
```

### Measured configurations

`python scripts/benchmark.py` on `data/test_video.mp4` (401 frames, 1280x720,
Windows 11, 16 logical CPUs). Annotated frames and video disabled, so the
end-to-end figure is free of JPEG encoding:

| Config | .pte MB | model avg ms | model p95 ms | model FPS | end-to-end avg ms | end-to-end FPS | peak RSS MB | mean person | mean vehicle_total | flagged frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 | 13.15 | 29.6 | 32.6 | 33.80 | 68.0 | 14.70 | 484 | 0.01 | 3.44 | 0 |
| **fp32 +tile** | 13.15 | 153.7 | 164.4 | 6.51 | 266.7 | 3.75 | 455 | **2.61** | 9.12 | 33 |
| fp32 +tile +io | 13.15 | 160.4 | 176.8 | 6.23 | 292.7 | 3.42 | 465 | 2.61 | 9.12 | 33 |
| int8 | 3.68 | 31.7 | 35.1 | 31.54 | 73.3 | 13.64 | 470 | 0.00 | 3.15 | 0 |
| int8 +tile | 3.68 | 162.2 | 174.3 | 6.16 | 290.4 | 3.44 | 471 | 1.96 | 7.84 | 10 |

`+io` marks the one run that also wrote 401 JPEGs and the video. Compare it
with the row above: writing files costs ~7 ms/frame of *model* time, not just
wall clock, because JPEG encoding competes with the XNNPACK thread pool. That
is why `--no-save-frames` exists and why every benchmark row uses it.

**Recommended default: `fp32 --tile`.**

- Tiling is not optional. Untiled finds **0.01 people per frame**; tiled finds
  **2.61**. A config that cannot see pedestrians makes the conflict flag
  meaningless, whatever its latency — the untiled rows flag 0 frames out of 401
  not because the crossing is safe but because the model never sees anybody on
  it. The cost is 5.2x the model time.
- FP32 over INT8 because INT8 finds **25% fewer people** (1.96 vs 2.61) and
  consequently flags 10 frames where FP32 flags 33, while being **no faster**
  (31.37 ms vs 31.42 ms in the interleaved micro-benchmark — a 0.2%
  difference, inside the noise). INT8's only win here is a 3.6x smaller file.
- **That calculus flips on a Pi**, where 13 MB of weights and ~450 MB peak RSS
  are a real constraint and INT8's memory footprint may be what makes the thing
  run at all. Re-measure there before choosing.

`--model eager` runs torchvision directly instead of the `.pte`. It is for
debugging and parity work, not for deployment.

---

## Outputs

Everything lands in `outputs/`.

| Path | What it is |
| --- | --- |
| `counts.csv` | one row per processed frame: `frame_idx, timestamp_s, person, bicycle, car, motorcycle, bus, truck, vehicle_total, persons_in_zone, vehicles_in_zone, conflict_flag` |
| `annotated/` | every processed frame with boxes, labels, scores, the zone polygon and a count panel |
| `annotated.mp4` | the same frames as a video |
| `samples/` | the 3 most informative frames (conflict first, then in-zone activity, then total detections) |
| `report.md` / `report.json` | model, precision, `.pte` size, peak memory, avg/p50/p95 ms (model-only *and* end-to-end), FPS, per-class mean/max/peak frame, conflict-frame count, FP32-vs-INT8 table, parity vs torchvision |
| `benchmark.md` / `benchmark.json` | model-only latency comparison |
| `bench_<precision>.json` | one record per run; this is what fills the FP32-vs-INT8 table |
| `eval_counts.md` / `.json` | counting accuracy vs your hand counts, with a threshold sweep |
| `detections.jsonl` | one JSON object per frame with every box, score and label — lets you re-tune the zone polygon and conflict thresholds offline, with no further inference |

`frame_idx` always indexes the **source**, before `--stride` subsampling, so
`--stride 5` produces rows 0, 5, 10… and a ground-truth file keeps meaning the
same thing across runs.

---

## Configuration

`config.yaml` holds everything you would otherwise edit code for.

```yaml
detection:
  score_threshold: 0.5      # default
  nms_iou: 0.45             # default
  classes: [person, bicycle, car, motorcycle, bus, truck]
  vehicle_classes: [bicycle, car, motorcycle, bus, truck]

zone:
  polygon_norm:             # normalized [0,1] -> resolution independent
    - [0.05, 0.47]
    - [0.95, 0.47]
    - [0.95, 0.66]
    - [0.05, 0.66]
  anchor_point: bottom_center

conflict:
  min_persons: 2
  min_vehicles: 1
  min_consecutive_frames: 5
```

**The zone polygon is scene geometry, not a constant.** The shipped one is
drawn for `data/test_video.mp4`. Re-draw it for your own camera: read the
vertices off one frame, divide by width and height.

`anchor_point: bottom_center` means a detection is "in the zone" when the
**bottom centre of its box** — the object's ground contact point — is inside
the polygon. Using the box centre would put a tall pedestrian standing safely
on the kerb inside a road zone.

### Calibrating the conflict flag

The flag is raised once

```
persons_in_zone >= min_persons  AND  vehicles_in_zone >= min_vehicles
```

has held for `min_consecutive_frames` **processed** frames in a row, and stays
raised while it keeps holding. The duration term counts processed frames, so it
scales with `--stride`: 5 frames is 0.17 s at stride 1 and 30 fps, but 0.83 s at
stride 5. The rule is causal — the first `min_consecutive_frames - 1` frames of
an episode are not flagged — so the annotated frame, the CSV row and a live
stream all agree without a second pass. Episode start times are recorded
separately and listed in `report.md`.

**Calibrate the zone before you touch the thresholds.** The first version of
this config covered the whole carriageway, and the result was useless: a flag on
**350/401 frames (87%)**. The cause was geometric, not statistical —
`vehicles_in_zone >= 3` was true in **99%** of frames, so the vehicle half of
the rule carried no information at all and raising `min_vehicles` just turned
the flag into a traffic-density counter. Tightening the polygon to the crosswalk
band itself made both terms discriminative:

| Zone | `frac(persons>=1)` | `frac(persons>=2)` | `frac(vehicles>=1)` | `frac(vehicles>=2)` |
| --- | --- | --- | --- | --- |
| old, whole carriageway (y 0.33–0.99) | 0.88 | 0.66 | **1.00** | **1.00** |
| crosswalk band (y 0.42–0.70) | 0.75 | 0.53 | 0.99 | 0.93 |
| **crosswalk, tight (y 0.47–0.66)** | 0.65 | **0.37** | **0.77** | 0.28 |

With the tight zone, at thresholds P=2 / V=1:

| rule | raw condition | flagged | episodes | mean episode |
| --- | --- | --- | --- | --- |
| P=1, V=1, D=5 | 47.6% | 17.2% | 10 | 0.36 s |
| **P=2, V=1, D=5 (default)** | 25.9% | **8.2%** | 4 | 0.41 s |
| P=2, V=1, D=3 | 25.9% | 12.2% | 10 | 0.23 s |
| P=2, V=1, D=8 | 25.9% | 5.5% | 3 | 0.48 s |
| P=2, V=2, D=3 | 5.5% | 1.5% | 2 | 0.17 s |

Why these three numbers:

- **`min_vehicles: 1`** — one vehicle inside the crossing while people are in it
  *is* the conflict. Raising it would measure congestion, not risk.
- **`min_persons: 2`** — guards against a single spurious `person` box. The
  duration rule cannot filter a false positive that is *persistent* (a bollard
  read as a person holds for hundreds of frames); a second independent person
  detection can. Note the detector finds roughly 1 in 10 of the people actually
  present on this clip, so "2 detected" is already a conservative reading.
- **`min_consecutive_frames: 5`** — 0.17 s at 30 fps, enough to drop
  single-frame detector flicker while keeping genuine crossings, which last
  0.2–0.7 s here.

The raw condition still holds on 25.9% of frames; the duration rule is what
takes that to 8.2%. Both numbers are in `report.md` so you can see what each
term is doing.

Re-tune per site. `outputs/detections.jsonl` caches every box, so you can sweep
zones and thresholds offline without paying for inference again.

---

## Measuring counting accuracy

```bash
# 1. export 10 frames and a ready-to-fill template
python scripts/eval_counts.py --export-frames 10

# 2. open outputs/gt_frames/*.jpg, count by hand,
#    fill the person and vehicle columns in outputs/gt_template.csv

# 3. score the model and sweep the score threshold 0.30 - 0.70
python scripts/eval_counts.py --gt outputs/gt_template.csv
python scripts/eval_counts.py --gt outputs/gt_template.csv --tile
```

The ground-truth format is `frame_idx,person,vehicle`, where `vehicle` is every
vehicle class added together (telling a van from a truck by eye is not
reliable, and the pipeline reports `vehicle_total` anyway).

Reported per threshold: **MAE**, **bias** (signed — negative means
under-counting), **RMSE**, **accuracy** (`1 - MAE / mean(GT)`, clipped at 0) and
the fraction of frames counted exactly right. The model runs once per frame;
the sweep re-runs only the NumPy post-processing, so nine thresholds cost
almost nothing.

This is **count accuracy, not detection accuracy** — no box matching happens,
so a frame with the right count in the wrong places scores perfectly. With ~10
labelled frames the error bars are wide. Treat it as a sanity check.

### What the sweep says on the sample clip

Run with `--tile` against 10 frames of `data/test_video.mp4`:

| threshold | person MAE | person bias | person acc | vehicle MAE | vehicle bias | vehicle acc |
| --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 38.5 | −38.5 | 0.16 | 1.1 | +0.3 | 0.89 |
| 0.35 | 40.1 | −40.1 | 0.13 | 1.0 | −0.2 | **0.90** |
| 0.50 (default) | 42.8 | −42.8 | 0.07 | 1.5 | −0.9 | 0.85 |
| 0.70 | 45.0 | −45.0 | 0.02 | 3.5 | −3.5 | 0.65 |

> The reference counts behind this table were estimated from the frames rather
> than counted carefully, so the absolute person figures carry perhaps ±25%.
> Re-count them before quoting any of this. The *conclusion* survives that
> error bar comfortably, and it is the important part:

**Vehicle counting works; pedestrian counting does not, on this clip.** Vehicles
land within ~1 of the truth (accuracy 0.85–0.90). Pedestrians do not: the clip
has 26–60 people per frame, mostly in a dense kerbside crowd, and the model
finds 2–9 of them even with tiling. The bias is large and negative at every
threshold, so no amount of threshold tuning fixes it — it is a recall ceiling,
not a calibration problem. Lowering the threshold to 0.30 buys ~4 extra people
per frame and costs vehicle precision.

For dense crowds this detector is the wrong tool: a 320×320 input, a crowd at
30–40 px per person, and COCO weights that never saw this viewpoint. It is
adequate for the sparser school-gate and bus-stop scenes SafeCross targets,
where pedestrians are nearer the camera and number in single digits — but that
claim is untested here and should be measured on footage from the actual site.

---

## How it is put together

```
export/export_model.py       pretrained model -> .pte (FP32 + INT8) + anchors + parity report
safecross/
  preprocess.py              BGR uint8 frame -> normalized 320x320 NCHW (NumPy only)
  runtime.py                 ExecuTorch .pte runtime; eager torch runtime for parity
  postprocess.py             anchor decode, softmax, threshold, class-aware NMS (NumPy only)
  counter.py                 detections -> per-class counts + vehicle_total
  zones.py                   polygon ROI, in-zone counting, conflict rule
  visualize.py               boxes, labels, zone overlay, count panel
  video.py                   video / image-folder frame source, video writer
scripts/
  run_pipeline.py            the pipeline + report generation
  benchmark.py               FP32 vs INT8 latency, size, memory
  eval_counts.py             ground-truth template export, MAE and threshold sweep
tests/                       pytest; 59 tests, no network, no GPU
config.yaml                  thresholds, zone polygon, conflict rule, output options
```

### What is inside the exported graph, and what is not

Only **backbone + SSD head** are exported, emitting raw `cls_logits`
`(1, 3234, 91)` and `bbox_regression` `(1, 3234, 4)`. The input transform,
anchor generation, softmax, score threshold, NMS and the box rescale all live
in NumPy.

That split is deliberate. NMS output is data-dependent, so putting it in the
graph forces dynamic shapes and blocks delegation to XNNPACK. What is left is
one static-shape convolutional network — exactly what a CPU backend wants —
and the parts that stay in NumPy are cheap, portable and testable.

The anchor table is **generated by torchvision's own `DefaultBoxGenerator`** at
export time and saved to `models/anchors.npy`. No anchor arithmetic is
re-derived by hand anywhere in this repo.

### Preprocessing matches torchvision exactly

`ssdlite320_mobilenet_v3_large` wraps the network in
`GeneralizedRCNNTransform(fixed_size=(320, 320))` with `mean=[0.5]*3`,
`std=[0.5]*3`, and the weights' transform only divides by 255. So the chain is:
BGR → RGB → `/255` → bilinear resize to 320×320 (`align_corners=False`, no
antialias, aspect ratio **not** preserved) → `(x - 0.5) / 0.5`.

The resize is implemented in NumPy using torch's own
`area_pixel_compute_source_index` formula, in float32, rather than
`cv2.resize` — `tests/test_preprocess.py` asserts it matches
`F.interpolate` to < 1e-5 and that the full preprocessing matches the real
`model.transform` to < 1e-4.

### Parity gate

`export/export_model.py` will not let you trust a `.pte` on faith. After each
export it runs the program and torchvision eager mode on the same frames and
compares both the raw tensors and the final detections, writing
`models/parity_<precision>.json`. Measured on `data/test_video.mp4`:

| Precision | max abs cls-logit diff | max abs bbox diff | detections matched vs eager |
| --- | --- | --- | --- |
| FP32 | 2.0e-05 | 3.7e-05 | 20 / 20 (1.000) |
| INT8 | 2.1e+00 | 5.1e+00 | 15 / 20 (0.750) |

FP32 is a lowering, not an approximation, and behaves like one. INT8 is
post-training quantization with no accuracy recovery, and the table says
exactly what that costs.

### Cross-class vehicle de-duplication

torchvision runs NMS **per class**, so one car can survive as both `car 0.52`
and `truck 0.70`. Harmless for a detection benchmark, but it double-counts
`vehicle_total`, and tiling makes it worse. SafeCross runs one extra
class-agnostic NMS restricted to the vehicle classes, on the argument that one
physical vehicle cannot be two vehicle types at once. People are never touched,
so a pedestrian standing against a car is never suppressed. Turn it off with
`detection.merge_duplicate_vehicles: false` to get raw torchvision behaviour.

---

## Raspberry Pi readiness

Nothing here is laptop-specific, but **nothing here has been run on a Pi yet**.
What has been done:

- CPU only. No CUDA import, no `.cuda()`, no GPU in the inference path.
- The deployment path is ExecuTorch + XNNPACK, which is the Pi path.
- Post-processing is NumPy, which has aarch64 wheels.
- `torch` is imported at inference only to hand tensors to the ExecuTorch
  Python bindings; that boundary is confined to `safecross/runtime.py`, so a
  C++ ExecuTorch runner can replace it without touching anything else.
- The `.pte` files are portable: the same files load on aarch64.

What is still unknown until it is measured on hardware: actual Pi latency,
whether 1 GB/2 GB models have enough headroom at ~450 MB peak RSS, and thermal
behaviour over a long recording. **Do not extrapolate the laptop numbers.**

---

## Honest limitations

- **No tracking.** Counts are per frame, not unique objects. A pedestrian in
  view for 100 frames is counted 100 times. Summing a CSV column is *not* a
  footfall total.
- **Stock COCO weights, no fine-tuning.** Small and distant pedestrians are the
  weakest case: at 320×320, a person 30 px tall in a 1280×720 frame is a
  handful of pixels after the resize. On the sample clip the untiled model
  finds essentially **no** pedestrians at threshold 0.5 (mean 0.02/frame) while
  counting vehicles reliably. `--tile` lifts that to 2.61/frame at roughly 5×
  the model cost — still far short of the 26–60 people actually present. See
  the sweep table above.
- **Class confusion among vehicles.** Taxis are frequently labelled `truck`.
  `vehicle_total` is trustworthy; the per-class split between `car`/`truck`/
  `bus` is not.
- **INT8 loses detections.** 3.6× smaller file, and on this x86 laptop **no
  measurable speed-up** (within 0.6% of FP32 when the two are interleaved to
  cancel thermal drift). Its value here is memory footprint, and that case is
  much stronger on a Pi than on a 16-core laptop.
- **The conflict flag is sustained co-presence, not risk.** It says a person and
  a vehicle shared a polygon for at least 0.17 s. It knows nothing about speed,
  distance, right of way or intent, and it cannot distinguish a parked car from
  one moving at 50 km/h. It is also only as good as the hand-drawn polygon: the
  87% → 8.2% fix above was a geometry fix, and a badly placed zone on new
  footage will produce an equally meaningless number.
- **The flag rate was tuned on the same clip it is reported for.** 8.2% is not
  a validated operating point, it is a calibration on 401 frames of one scene
  with no labelled conflicts to check against. There is no ground truth for
  "was this actually a conflict", so the thresholds are argued from the count
  distributions, not measured against outcomes.
- **The zone is hand-drawn and static.** A camera nudge invalidates it. There
  is no homography, so "in the zone" is an image-plane test, not a
  ground-plane one.
- **Timings are from one laptop** (Windows 11, 16 logical CPUs), single
  process, batch size 1. The end-to-end figure includes JPEG encoding of every
  annotated frame, which dominates at small frame counts.
- **The accuracy evaluation uses ~10 hand-labelled frames.** That is enough to
  catch a gross error, not enough to quote a precision figure.

---

## Tests

```bash
python -m pytest
```

59 tests, no network access, no GPU. The parity tests skip cleanly on a fresh
clone where `models/*.pte` does not exist yet, so the suite passes before the
first export.

---

## Sample data

`data/test_video.mp4` is a busy signalised crossing (pedestrian crowd, taxis, a
coach, vans). See `data/SOURCE.md` for its origin and licence. Replace it with
your own footage — nothing in the repo depends on that particular clip, though
you will want to re-draw `zone.polygon_norm` for a different scene.
