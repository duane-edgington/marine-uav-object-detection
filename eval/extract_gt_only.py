#!/usr/bin/env python3
"""
extract_gt_only.py -- extract just the ground-truth annotations (no model predictions) from a
detections_bgNN.json file, into a clean, standalone CSV.

GT is identical across every detections_bgNN.json in a given pipeline run (only the model's
predictions differ across background levels) -- this pulls it out of whichever file you point it
at, verifies the total box count, and writes a plain CSV with no prediction data mixed in.

Run:
  python extract_gt_only.py --det-file model_compare/native_v1_gtfixed/detections_bg00.json \
    --out ground_truth_98image_benchmark.csv
"""
import csv
import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-file', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.det_file) as f:
        d = json.load(f)
    records = d['records']

    rows = []
    for r in records:
        for x1, y1, x2, y2 in r['gt']['xyxy']:
            rows.append((r['image'], x1, y1, x2, y2))

    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_image', 'x1', 'y1', 'x2', 'y2'])
        writer.writerows(rows)

    print(f'{len(records)} images, {len(rows)} total GT boxes')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
