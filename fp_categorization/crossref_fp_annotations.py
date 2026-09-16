#!/usr/bin/env python3
"""
crossref_fp_annotations.py -- find the SAME real-world FP location appearing across multiple
background-level manifests (same source image, close box centers), then:
  1. Report where independently-assigned categories AGREE vs DISAGREE (consistency check)
  2. For any manifest with blank "category" cells, suggest a category pulled from a matching
     row in an already-annotated manifest -- SUGGESTIONS ONLY, never silently overwrites a
     human's judgment, and never overwrites an already-filled cell.

Matching is by (source_image, center-distance < --match-dist), not IoU -- different models draw
somewhat different box sizes/positions around the same real-world feature, so requiring tight
IoU overlap would miss genuine matches. Center-distance is a looser, more robust criterion for
"this is probably the same physical spot."

Run (all already-categorized manifests as references, one target to check/pre-fill):
  python crossref_fp_annotations.py --manifests 0bg_manifest.csv 10bg_manifest.csv 20bg_manifest.csv \
    --target 5bg_manifest.csv --out-suggestions 5bg_manifest_suggested.csv \
    --out-consistency consistency_report.csv
"""
import csv
import argparse
from collections import defaultdict


def load_manifest(path, label):
    with open(path, encoding='utf-8-sig') as f:
        lines = f.readlines()
    start = 1 if ',' not in lines[0] else 0  # skip stray Numbers sheet-name line if present
    reader = csv.DictReader(lines[start:])
    rows = list(reader)
    for r in rows:
        r['_source_label'] = label
        r['_cx'] = (float(r['x1']) + float(r['x2'])) / 2
        r['_cy'] = (float(r['y1']) + float(r['y2'])) / 2
    return rows


def dist(r1, r2):
    return ((r1['_cx'] - r2['_cx']) ** 2 + (r1['_cy'] - r2['_cy']) ** 2) ** 0.5


def find_matches(rows_a, rows_b, match_dist):
    """For each row in rows_a, find all rows in rows_b on the same source image within match_dist."""
    by_image = defaultdict(list)
    for r in rows_b:
        by_image[r['source_image']].append(r)
    matches = {}
    for i, ra in enumerate(rows_a):
        candidates = by_image.get(ra['source_image'], [])
        best = None
        best_d = match_dist
        for rb in candidates:
            d = dist(ra, rb)
            if d < best_d:
                best, best_d = rb, d
        if best is not None:
            matches[i] = (best, best_d)
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifests', nargs='+', required=True,
                     help='already-categorized manifest CSVs to cross-reference and use as suggestion sources')
    ap.add_argument('--target', default=None,
                     help='a manifest to pre-fill suggestions for (blank category cells only) -- optional')
    ap.add_argument('--out-suggestions', default='suggested_manifest.csv')
    ap.add_argument('--out-consistency', default='consistency_report.csv')
    ap.add_argument('--match-dist', type=float, default=75.0,
                     help='max center-to-center pixel distance to count as "the same real-world spot"')
    args = ap.parse_args()

    labeled = {}
    for path in args.manifests:
        label = path.replace('_manifest.csv', '').replace('.csv', '')
        labeled[label] = load_manifest(path, label)

    # --- Consistency check: for every pair of already-categorized manifests, find matched
    # locations and report where their independently-assigned categories agree or disagree ---
    labels = list(labeled.keys())
    consistency_rows = []
    n_agree, n_disagree = 0, 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            matches = find_matches(labeled[la], labeled[lb], args.match_dist)
            for idx_a, (row_b, d) in matches.items():
                row_a = labeled[la][idx_a]
                cat_a, cat_b = row_a['category'].strip(), row_b['category'].strip()
                agree = (cat_a == cat_b)
                n_agree += agree
                n_disagree += not agree
                consistency_rows.append({
                    'source_image': row_a['source_image'],
                    'manifest_A': la, 'category_A': cat_a, 'notes_A': row_a.get('Notes', ''),
                    'manifest_B': lb, 'category_B': cat_b, 'notes_B': row_b.get('Notes', ''),
                    'center_distance_px': f'{d:.1f}',
                    'agree': 'YES' if agree else 'NO -- CHECK THIS ONE',
                })

    with open(args.out_consistency, 'w', newline='') as f:
        fieldnames = ['source_image', 'manifest_A', 'category_A', 'notes_A',
                      'manifest_B', 'category_B', 'notes_B', 'center_distance_px', 'agree']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # disagreements first -- these are the ones worth actually looking at
        for row in sorted(consistency_rows, key=lambda r: r['agree'] != 'YES', reverse=True):
            writer.writerow(row)

    print(f'Consistency check across {labels}: {len(consistency_rows)} matched locations found '
          f'(center distance < {args.match_dist}px)')
    print(f'  agree: {n_agree}   disagree: {n_disagree}')
    print(f'  Full report (disagreements listed first): {args.out_consistency}')

    # --- Pre-fill suggestions for a target manifest, if given ---
    if args.target:
        target_rows = load_manifest(args.target, 'target')
        all_ref_rows = [r for lbl in labels for r in labeled[lbl]]
        matches = find_matches(target_rows, all_ref_rows, args.match_dist)

        n_blank = sum(1 for r in target_rows if not r['category'].strip())
        n_suggested = 0
        for idx, row in enumerate(target_rows):
            if row['category'].strip():
                continue  # never touch an already-filled cell
            if idx in matches:
                ref_row, d = matches[idx]
                row['category'] = ''  # leave the real column blank -- human still decides
                row['suggested_category'] = ref_row['category'].strip()
                row['suggested_from'] = f"{ref_row['_source_label']} (dist={d:.0f}px)"
                if ref_row.get('Notes', '').strip():
                    row['suggested_notes'] = ref_row['Notes'].strip()
                n_suggested += 1

        out_fields = ['crop_filename', 'source_image', 'x1', 'y1', 'x2', 'y2', 'confidence',
                      'bg_level', 'category', 'suggested_category', 'suggested_from',
                      'suggested_notes', 'Notes']
        with open(args.out_suggestions, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
            writer.writeheader()
            for row in target_rows:
                for fld in out_fields:
                    row.setdefault(fld, '')
                writer.writerow(row)

        print(f'\nTarget manifest "{args.target}": {len(target_rows)} rows, {n_blank} blank.')
        print(f'  {n_suggested} of {n_blank} blanks got a suggestion from an already-annotated match.')
        print(f'  {n_blank - n_suggested} blanks have no match in the reference manifests -- fresh judgment needed.')
        print(f'  Saved: {args.out_suggestions} (suggestions are in their own column -- category stays blank until you confirm)')


if __name__ == '__main__':
    main()
