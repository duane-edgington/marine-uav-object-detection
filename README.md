# marine-uav-object-detection

Evaluation code for **"Background Augmentation for Reliable Marine UAV Object Detection,"**
submitted to the IEEE Underwater Technology Symposium 2027 (Remote Sensing focus area).

This repository contains the evaluation and false-positive-categorization code developed for
this work. It accompanies the paper above.

**What's here**: evaluation code (confidence-threshold sweeps, F1/precision-recall curves, false
positives by category) and the underlying results it produced. **What's not here**: training
code (placeholder included — see below) and the raw images/trained models, which are hosted
elsewhere (see Links, below) to keep this repository lightweight and focused on reproducible
evaluation.

## On our choice of evaluation metric

**Our primary metrics are NWD (Normalized Wasserstein Distance) and DotD (Dot Distance)**, both
designed for tiny-object detection, where standard IoU is known to be disproportionately
punishing toward small localization errors on small objects (a fixed few-pixel offset that is
negligible on a large object can collapse IoU to near zero on a small one). We report
**IoU-based mAP50 as well, purely as a courtesy to readers expecting a standard COCO-style
metric** — it is not the metric our conclusions are built on, and should not be read as our
primary evidence.

## Data and models

- **Trained model checkpoints**: [Hugging Face — link to be added]
- **Benchmark images**: already public; this submission does not emphasize dataset construction,
  so we point to it rather than re-host it here — [link to be added]
- **Ground truth annotations**: `eval/extract_gt_only.py` extracts a clean, standalone copy from
  any detections JSON (see below) — every detections file carries an identical copy of GT, since
  only model predictions differ across background-augmentation levels.

## Known limitation: checkpoint selection

We have found, in both architectures evaluated here, that the checkpoint a training framework
selects as "best" is not always the checkpoint that performs best under our full tiled-inference
evaluation. This is documented in detail, including what we have and have not yet investigated,
in **`docs/checkpoint_selection_limitation.md`**. We are releasing results using the
framework-selected checkpoints (not hand-picked-after-the-fact ones) and flagging this
transparently rather than waiting for that investigation to complete.

## Pipeline overview

```
trained model + benchmark images
        |
        v
  [inference -- SAHI-style tiled detection, not included here; see model repo]
        |
        v
  detections_bgNN.json   (per-image predictions + ground truth, full-image coordinates)
        |
        v
  eval/fix_gt_shift.py   (ground-truth coordinate correction -- see "Known data issue" below)
        |
        v
  eval/nwd_report.py, eval/dotd_report.py   (primary metrics)
  eval/report_local.py                      (IoU/mAP50, for COCO-style comparison)
        |
        v
  confidence-threshold sweeps -> F1 curves, precision-recall curves, false-positive-vs-confidence
        |
        v
  fp_categorization/*.py  -> false-positive-by-category analysis (see docs/FP_categorization_summary.md)
```

## Detections JSON format

All evaluation scripts consume a common JSON format, produced by our tiled inference pipeline:

```json
{
  "meta": {"model": "...", "bg_percent": 0, "pipeline": "...", "n_images": 98, ...},
  "records": [
    {
      "image": "example.JPG",
      "gt":   {"xyxy": [[x1,y1,x2,y2], ...], "confidence": [], "class_id": [...]},
      "pred": {"xyxy": [[x1,y1,x2,y2], ...], "confidence": [...], "class_id": [...]}
    }
  ]
}
```

Coordinates are in full-image pixel space (not tile-local) — a single class ("object") is used
throughout. `gt.confidence` is always an empty list (ground truth has no confidence value); this
is expected and handled by every script here.

## Known data issue: ground-truth coordinate correction

An early version of our data export pipeline (`mbari_aidata`'s `coco_voc.py`) wrote a bounding
box's top-left corner directly into fields expected to hold the box *center*, without adding
half the width/height. This silently shifted every ground-truth box by half its own width and
height. **`eval/fix_gt_shift.py` corrects this** — run it on any raw detections JSON before using
any of the other `eval/` scripts. All results in our paper and in this repository use
GT-corrected data; the fix has been merged upstream into `mbari_aidata`. If you are using a
detections JSON that has already had this correction applied, running `fix_gt_shift.py` again on
already-corrected data will silently double-shift it — check your file's `meta.gt_correction_applied`
field, or simply confirm which stage of the pipeline your file came from, before running it.

## False-positive definition (used throughout eval/ and fp_categorization/)

A false positive is a prediction (at or above a given confidence threshold) that, under
**one-to-one, confidence-ordered greedy matching** — predictions considered highest-confidence
first, each ground-truth box claimable by at most one prediction — has no ground-truth box within
the metric's matching threshold (e.g. IoU >= 0.1, or the corresponding NWD/DotD threshold). This
is *not* the same as checking each prediction independently against all ground truth: independent
checking would incorrectly treat duplicate detections of the same real object as all "matched,"
undercounting false positives whenever a detector fires more than once on one object. See
`fp_categorization/export_fps_for_voxel51.py`'s `match_image()` for the reference implementation,
and its `max_iou_vs_matched_TP` output column for a way to distinguish this kind of
duplicate-detection artifact (a Non-Maximum-Suppression/post-processing issue) from a genuine
false positive.

## Directory guide

- **`eval/`** — core scoring: confidence sweeps, F1 curves, precision-recall curves, false
  positives vs. confidence, and peak-F1 summaries, under NWD, DotD, and (for reader convenience)
  IoU/mAP50. See `eval/README.md`.
- **`fp_categorization/`** — tools for categorizing false positives by underlying cause (wave
  texture, glare, missed real objects, etc.), including a red-box-annotated crop export for
  manual review and a cross-manifest consistency checker. See `fp_categorization/README.md`.
- **`results/manifests/`** — the actual, complete, hand-categorized false-positive manifests for
  all four RF-DETR background-augmentation levels evaluated in the paper (0%, 5%, 10%, 20%).
- **`results/figures/`** — three progressively-expanded comparison figure sets: RF-DETR alone
  across all five background levels (0/5/10/20/40%), the same plus YOLO26 at 0% background, and
  the same plus YOLO26 at 5% background.
- **`docs/FP_categorization_summary.md`** — full methodology and findings from the false-positive
  categorization analysis.
- **`docs/checkpoint_selection_limitation.md`** — the checkpoint-selection limitation described
  above, in full.

## Requirements

```
numpy
matplotlib
pillow
```

No deep-learning framework (PyTorch, etc.) is required for anything in this repository — all
evaluation here operates on already-computed detection JSONs, not on models directly. Generating
those JSONs in the first place requires the trained models and an inference environment; see the
model repository (Hugging Face link above) for that half of the pipeline.

## Training code

Not included in this initial release. If useful to reviewers or other researchers, we plan to
add training scripts (RF-DETR and YOLO26 configurations used in this work) in a future update —
open an issue if this would be useful to you sooner.

## Citation

```
[Author list]. "Background Augmentation for Reliable Marine UAV Object Detection."
IEEE Underwater Technology Symposium (UT), 2027. [DOI/link to be added upon publication]
```

## License

Apache License 2.0 — see `LICENSE`.
