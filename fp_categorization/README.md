# `fp_categorization/` — false-positive categorization tools

Full methodology and results: see `../docs/FP_categorization_summary.md`. This directory holds
the tools that produced that analysis.

## `crop_false_positives.py`

Identifies every false positive at a given confidence threshold (using the one-to-one matching
described in the top-level README), then samples a manageable, capped-per-image subset and
crops each one out of the full-resolution source image — with generous padding for background
context, and the exact detection box drawn as a red outline for unambiguous localization. Writes
a manifest CSV with a blank `category` column for hand-labeling.

```bash
python crop_false_positives.py --det-file detections_bg00_gtfixed.json \
  --images-dir /path/to/images --out fp_crops --label 0bg --conf 0.5 --sample-n 200
```

Sampling is capped at 5 crops per source image (`--max-per-image`) so a single cluttered frame
can't dominate the sample, then randomly sampled down to `--sample-n` total (fixed seed, so the
sample is reproducible).

## `export_fps_for_voxel51.py`

Exhaustive (not sampled) export of every false positive at a given confidence threshold, as a
plain CSV — no image cropping, just coordinates and metadata, intended for import into external
tools (e.g. Voxel51) for dataset curation and re-annotation.

Includes an `max_iou_vs_matched_TP` column: for each false positive, its IoU overlap against the
nearest *already-correctly-matched* detection on the same image. A high value here indicates the
false positive is likely a duplicate/fragment detection of an object the model already found
correctly (a Non-Maximum-Suppression / post-processing artifact — NMS only suppresses heavily-
overlapping boxes, so a thin fragment that only partially overlaps a correct detection survives
as a separate false positive) — as opposed to a genuine annotation gap or model hallucination.

```bash
python export_fps_for_voxel51.py --det-file detections_bg00_gtfixed.json \
  --conf 0.3 --label 0bg --out fp_export_0bg.csv
```

## `crossref_fp_annotations.py`

Two functions, both operating on a set of already-hand-categorized manifest CSVs (as produced by
`crop_false_positives.py` once you've filled in the `category` column):

1. **Consistency check**: since the same real-world false-positive location often recurs across
   multiple background-level samples (all models are evaluated on the same images), this finds
   matching locations (same source image, box centers within a configurable pixel distance) and
   reports where independently-assigned categories agree or disagree.
2. **Suggestion pre-fill**: for a manifest you haven't categorized yet, suggests categories for
   any location that matches something already categorized in a reference manifest — speeding up
   repeat annotation work without ever overwriting your own judgment (suggestions are written to
   a separate column; the real `category` column stays blank until you confirm it yourself).

```bash
python crossref_fp_annotations.py --manifests 0bg_manifest.csv 10bg_manifest.csv 20bg_manifest.csv \
  --out-consistency consistency_report.csv \
  --target 5bg_manifest_blank.csv --out-suggestions 5bg_manifest_suggested.csv
```

## Category codes used in `results/manifests/`

| Code | Meaning |
|---|---|
| S | Shadow |
| H | Human (includes associated gear/vessels: kayak, boat, surfboard) |
| E | Excreta (bright white dots; visually similar to glint/foam; likely mostly liquid bird droppings) |
| B | Bird |
| W | Wave |
| G | Glint/Glare |
| F | Foam |
| K | Kelp (coarse — see Km/Kf) |
| Km | Kelp — mat/raft |
| Kf | Kelp — fragment |
| Wk | Wake |
| O | Other (coarse — see Od/Oc) |
| Od | Other — too dark/blurred/submerged to identify |
| Oc | Other — clearly visible, doesn't fit any category (includes real marine megafauna found in
     this dataset's false positives: whale, shark, jelly, fish, pinniped, otter, mola, bat ray) |
| N | Empty — no visible feature at all under the box |
