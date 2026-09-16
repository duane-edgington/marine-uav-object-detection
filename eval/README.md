# `eval/` — core scoring scripts

## `fix_gt_shift.py`

Corrects the ground-truth coordinate bug described in the top-level README. Run this first,
on the raw detections JSON, before anything else.

```bash
python fix_gt_shift.py --det-file detections_bg00.json --out detections_bg00_gtfixed.json
```

## `extract_gt_only.py`

Pulls just the ground-truth annotations (no predictions) out of a detections JSON, into a plain
CSV. Useful for sharing ground truth independently, or for a quick sanity check that GT is
identical across every background-level file in a given run (it should be — only predictions
differ).

```bash
python extract_gt_only.py --det-file detections_bg00_gtfixed.json --out ground_truth.csv
```

## `nwd_report.py`

**Primary metric.** Matches using Normalized Wasserstein Distance (NWD) — treats each box as a
small 2D Gaussian and measures distance between the ground-truth and predicted distributions,
scaled by a dataset-wide constant `C` (by default, the median ground-truth √-area — this makes
"how close is close enough" adapt to typical object size rather than being fixed in raw pixels).
NWD is less punishing than IoU toward small localization errors on small objects — see
`../docs/FP_categorization_summary.md` and the paper for the full motivation.

```bash
python nwd_report.py --det-dir /path/to/gtfixed_detections/
```

`C` is printed at the end of the run — report it alongside any NWD-based numbers, since it's
dataset-specific (computed from your own benchmark's object-size distribution, not a universal
constant).

## `dotd_report.py`

**Primary metric**, alongside NWD. Dot Distance (Xu, Wang, Yang, Yu, CVPRW 2021) — a simpler,
purely center-distance similarity metric: `DotD = exp(-D/S)`, where `D` is the Euclidean distance
between predicted and ground-truth box centers, and `S` is the RMS average ground-truth object
size for the benchmark (printed at the end of the run, same reporting convention as NWD's `C`).
Unlike NWD, DotD ignores box width/height entirely — a useful second, independent check: if a
finding holds under both NWD (which does encode box shape) and DotD (which doesn't), that's
stronger evidence the finding isn't an artifact of the shape-modeling choice specifically.

```bash
python dotd_report.py --det-dir /path/to/gtfixed_detections/
```

Auto-discovers every `detections_bg*.json` file in `--det-dir`, same as `report_local.py` and
`nwd_report.py` — no need to list background levels explicitly.

## `report_local.py`

**IoU-based scoring, included as a courtesy for readers expecting standard COCO-style metrics —
not our primary evidence (see the top-level README).** Computes a confidence sweep, IoU-threshold
sweep, false-positives-vs-confidence, F1-vs-confidence, precision-recall curves, confusion
matrices, and a compact mAP50/mAP50-95 summary. Auto-discovers every `detections_bg*.json` file
in a directory and treats each as a separate model/condition to compare.

```bash
python report_local.py --det-dir /path/to/gtfixed_detections/ --out-dir report_output/
```

Produces (in `--out-dir`): `conf_sweep.csv`, `iou_sweep.csv`, `model_summary.csv` (the mAP50 /
mAP50-95 table), and the corresponding PNG figures (`fp_vs_conf*.png`, `f1_vs_conf*.png`,
`pr_curves*.png`, `confusion_matrices*.png`).
