#!/usr/bin/env python3
"""
fix_gt_shift.py -- correct the confirmed systematic GT placement bug in existing detections_bg*.json
files, WITHOUT needing to re-download from TATOR or touch the original .txt label files.

CONFIRMED ROOT CAUSE (2026-07-29): mbari_aidata/generators/coco_voc.py's download() function
writes TATOR's native top-left (x, y) directly into the YOLO .txt file's center-x/center-y
fields, omitting the required +width/2, +height/2 conversion. Proven exactly against a manually
re-derived true box: the shifted box's CENTER exactly equals the TRUE box's TOP-LEFT CORNER, and
the shift magnitude exactly equals half the box's own width/height (not a fixed pixel offset).

CORRECTION FORMULA (derived from the proof above, operates directly on already-stored pixel
xyxy, no need to re-parse the original .txt files):
  given shifted box [x1_s, y1_s, x2_s, y2_s]:
    w = x2_s - x1_s
    h = y2_s - y1_s
    x1_true = (x1_s + x2_s) / 2
    y1_true = (y1_s + y2_s) / 2
    x2_true = x1_true + w
    y2_true = y1_true + h

Run:
  python fix_gt_shift.py --det-file model_compare/native_v1_from_spark/detections_bg00.json \
    --out model_compare/native_v1_from_spark/detections_bg00_gtfixed.json
"""
import json, argparse

def fix_box(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1_true = (x1 + x2) / 2
    y1_true = (y1 + y2) / 2
    return [x1_true, y1_true, x1_true + w, y1_true + h]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-file', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    d = json.load(open(args.det_file))
    n_boxes = 0
    for r in d['records']:
        r['gt']['xyxy'] = [fix_box(b) for b in r['gt']['xyxy']]
        n_boxes += len(r['gt']['xyxy'])

    d.setdefault('meta', {})['gt_correction_applied'] = (
        'Corrected systematic GT shift bug (mbari_aidata/generators/coco_voc.py download() -- '
        'TATOR top-left x,y written directly as YOLO center x,y with no +w/2,+h/2 conversion). '
        'Fix: x1_true=(x1_s+x2_s)/2, y1_true=(y1_s+y2_s)/2, x2_true=x1_true+w, y2_true=y1_true+h.'
    )
    with open(args.out, 'w') as f:
        json.dump(d, f)
    print(f'Corrected {n_boxes} GT boxes across {len(d["records"])} images -> {args.out}')

if __name__ == '__main__':
    main()
