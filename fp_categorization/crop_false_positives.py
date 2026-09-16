#!/usr/bin/env python3
"""
crop_false_positives.py -- extract image crops of false-positive detections, for manual (or
automatic) categorization by background type (wave texture, glare, foam, kelp, wake, other).

Built in response to reviewer x65Y's comment: "it would help to know which kinds of background
drive the false positives." Uses the SAME false-positive definition as the paper's own Table 1
(IoU >= 0.1 matching, confidence >= 0.5) so results stay directly comparable to numbers already
in the paper -- not a new/different FP definition.

A "false positive" here = a prediction (at conf >= --conf) whose IoU against every ground-truth
box in that image is below --iou-thresh. Each FP gets cropped with generous padding (not just the
tight box) since categorizing BACKGROUND TYPE requires seeing surrounding context, not just the
box itself -- especially important given many objects here are only tens of pixels wide.

To keep human categorization tractable (a "few hundred crops per model" per the reviewer's own
suggestion, not several thousand), this samples down: caps crops per source image (avoids one
cluttered frame dominating the sample) then randomly samples to --sample-n total, with a fixed
seed for reproducibility.

Outputs, per background level:
  <out>/<bg_label>/crop_0001_<image_stem>_conf0.XX.jpg  (crops, generous padding)
  <out>/<bg_label>_manifest.csv  (crop filename, source image, box coords, confidence, and a
                                   blank "category" column ready to fill in by hand)

Run:
  python crop_false_positives.py --det-file model_compare/native_v1_gtfixed/detections_bg00.json \
    --images-dir benchmark/test/images --out fp_crops_0bg --label 0bg --conf 0.5 --sample-n 200
"""
import os
import csv
import json
import random
import argparse
from PIL import Image, ImageDraw


def iou(boxA, boxB):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    ix1, iy1 = max(xA1, xB1), max(yA1, yB1)
    ix2, iy2 = min(xA2, xB2), min(yA2, yB2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def find_false_positives(gt_boxes, preds, iou_thresh):
    """Identical matching algorithm to report_local.py's match_image(): predictions sorted by
    confidence, descending; each GT box claimable by at most one prediction. Returns the subset
    of preds that are FPs under this one-to-one scheme -- NOT just "no GT overlaps at all,"
    which would incorrectly exclude duplicate/redundant detections near an already-claimed GT."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])  # sort by confidence desc
    claimed_gt = set()
    fps = []
    for i in order:
        pxy, pconf = preds[i]
        best_iou, best_g = 0.0, -1
        for g, gbox in enumerate(gt_boxes):
            if g in claimed_gt:
                continue
            v = iou(pxy, gbox)
            if v > best_iou:
                best_iou, best_g = v, g
        if best_iou >= iou_thresh:
            claimed_gt.add(best_g)  # this pred is the TP for that GT; not an FP
        else:
            fps.append((pxy, pconf))  # no unclaimed GT close enough -> FP
    return fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-file', required=True, help='a detections_bgNN.json file (GT-corrected)')
    ap.add_argument('--images-dir', required=True, help='dir containing the actual JPEGs')
    ap.add_argument('--out', required=True, help='output dir for crops + manifest')
    ap.add_argument('--label', required=True, help='short tag for this background level, e.g. "0bg"')
    ap.add_argument('--conf', type=float, default=0.5, help='confidence threshold, matches Table 1 (0.5)')
    ap.add_argument('--iou-thresh', type=float, default=0.1,
                     help='a prediction is an FP if its best IoU vs any GT is below this -- matches Table 1 (0.1)')
    ap.add_argument('--pad-factor', type=float, default=3.0,
                     help='crop padding as a multiple of the box\'s own size on each side, for background context')
    ap.add_argument('--min-pad-px', type=int, default=150,
                     help='minimum padding in pixels, so even tiny boxes get a meaningfully large crop')
    ap.add_argument('--max-per-image', type=int, default=5,
                     help='cap FP crops taken from any single image, so one cluttered frame cannot dominate the sample')
    ap.add_argument('--sample-n', type=int, default=200,
                     help='randomly sample down to this many total crops after the per-image cap')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    crop_dir = os.path.join(out_dir, args.label)
    os.makedirs(crop_dir, exist_ok=True)

    with open(os.path.expanduser(args.det_file)) as f:
        d = json.load(f)
    records = d['records']

    # Step 1: find every FP across the whole dataset, grouped by source image
    fps_by_image = {}
    n_total_preds_at_conf = 0
    for r in records:
        gt_boxes = r['gt']['xyxy']
        preds = [(xy, c) for xy, c in zip(r['pred']['xyxy'], r['pred']['confidence']) if c >= args.conf]
        n_total_preds_at_conf += len(preds)
        image_fps = find_false_positives(gt_boxes, preds, args.iou_thresh)
        if image_fps:
            fps_by_image[r['image']] = image_fps

    n_fp_total = sum(len(v) for v in fps_by_image.values())
    print(f'{args.label}: {n_total_preds_at_conf} predictions at conf>={args.conf}, '
          f'{n_fp_total} are false positives (IoU<{args.iou_thresh} vs all GT)')

    # Step 2: cap per image, then flatten to one pool
    pool = []  # (image_name, box_xyxy, confidence)
    for image_name, fps in fps_by_image.items():
        capped = fps if len(fps) <= args.max_per_image else random.sample(fps, args.max_per_image)
        for pxy, pconf in capped:
            pool.append((image_name, pxy, pconf))
    print(f'  after capping at {args.max_per_image}/image: {len(pool)} candidates')

    # Step 3: sample down to the target total
    if len(pool) > args.sample_n:
        pool = random.sample(pool, args.sample_n)
    print(f'  final sample: {len(pool)} crops')

    # Step 4: crop with generous padding and write the manifest
    manifest_path = os.path.join(out_dir, f'{args.label}_manifest.csv')
    with open(manifest_path, 'w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(['crop_filename', 'source_image', 'x1', 'y1', 'x2', 'y2',
                          'confidence', 'bg_level', 'category'])

        image_cache = {}
        for i, (image_name, (x1, y1, x2, y2), pconf) in enumerate(pool):
            if image_name not in image_cache:
                img_path = os.path.join(os.path.expanduser(args.images_dir), image_name)
                if not os.path.exists(img_path):
                    print(f'  SKIP missing image: {image_name}')
                    continue
                image_cache[image_name] = Image.open(img_path).convert('RGB')
            im = image_cache[image_name]
            W, H = im.size

            bw, bh = x2 - x1, y2 - y1
            pad_x = max(bw * args.pad_factor, args.min_pad_px)
            pad_y = max(bh * args.pad_factor, args.min_pad_px)
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(W, int(x2 + pad_x))
            cy2 = min(H, int(y2 + pad_y))

            crop = im.crop((cx1, cy1, cx2, cy2))
            crop = crop.copy()  # ensure a mutable image for drawing
            draw = ImageDraw.Draw(crop)
            # convert original-image box coords to crop-local coords, draw a visible outline
            # so it's unambiguous which specific feature is the one being categorized
            box_local = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
            draw.rectangle(box_local, outline=(255, 0, 0), width=max(2, int(min(bw, bh) * 0.04)))
            stem = os.path.splitext(image_name)[0]
            crop_fname = f'crop_{i+1:04d}_{stem}_conf{pconf:.2f}.jpg'
            crop.save(os.path.join(crop_dir, crop_fname), quality=90)
            writer.writerow([crop_fname, image_name, f'{x1:.0f}', f'{y1:.0f}', f'{x2:.0f}', f'{y2:.0f}',
                              f'{pconf:.3f}', args.label, ''])

        # free memory as we go for large runs
        image_cache.clear()

    print(f'Saved {len(pool)} crops to {crop_dir}')
    print(f'Manifest (fill in the "category" column by hand): {manifest_path}')


if __name__ == '__main__':
    main()
