#!/usr/bin/env python3
"""
nwd_report.py — re-evaluate EXISTING detections under Normalized Wasserstein Distance (NWD),
a similarity metric designed for tiny-object detection (Wang, Xu, Yang & Yu 2021, arXiv:2110.13389;
Xu et al. 2022, ISPRS J. Photogrammetry and Remote Sensing 190:79-93), as an alternative to the
IoU>=0.5 convention that is known to collapse for small objects with any location deviation.

NO NEW INFERENCE NEEDED: reads the same detections_bg*.json files already produced by
rf_detr_multimodel_bg_compare_LOCAL.py / rf_detr_native_slicer_compare.py.

NWD formula (boxes as cx,cy,w,h):
    W2 = (cxa-cxb)^2 + (cya-cyb)^2 + (wa/2-wb/2)^2 + (ha/2-hb/2)^2
    NWD = exp( -sqrt(W2) / C )
C is a dataset-dependent constant; default here is the MEDIAN ground-truth sqrt-area of this
benchmark (see --c-const to override, or --c-auto to recompute from the loaded GT).

Sweeps NWD "match threshold" the same way the existing report sweeps IoU, so results are directly
comparable to report_local.py's IoU-based table.

Run:
  python nwd_report.py --det-dir ~/rfdetr-work/model_compare/sarah_clean_v3
"""
import os, json, argparse, csv, glob
import numpy as np

def xyxy_to_cxcywh(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)

def nwd(a_cxcywh, b_cxcywh, C):
    cxa,cya,wa,ha = a_cxcywh
    cxb,cyb,wb,hb = b_cxcywh
    w2 = (cxa-cxb)**2 + (cya-cyb)**2 + (wa/2-wb/2)**2 + (ha/2-hb/2)**2
    return float(np.exp(-np.sqrt(w2)/C))

def match_image_nwd(gt_xyxy, pred_xyxy, pred_conf, nwd_thr, C):
    order = np.argsort(-np.asarray(pred_conf)) if len(pred_conf) else np.array([], int)
    preds = [pred_xyxy[i] for i in order] if len(pred_xyxy) else []
    gts = [xyxy_to_cxcywh(g) for g in gt_xyxy]
    preds_c = [xyxy_to_cxcywh(p) for p in preds]
    matched = set()
    tp = [False]*len(preds_c)
    for j, p in enumerate(preds_c):
        best_g, best_s = -1, -1.0
        for gi, g in enumerate(gts):
            if gi in matched: continue
            s = nwd(g, p, C)
            if s > best_s:
                best_s, best_g = s, gi
        if best_g >= 0 and best_s >= nwd_thr:
            tp[j] = True
            matched.add(best_g)
    return sum(tp), len(preds_c)-sum(tp), len(gts)-len(matched)

def confusion_nwd(records, conf_thr, nwd_thr, C):
    tp=fp=fn=0
    for r in records:
        keep = [i for i,c in enumerate(r['pred_conf']) if c >= conf_thr]
        pxy = [r['pred_px'][i] for i in keep]
        pcf = [r['pred_conf'][i] for i in keep]
        t,f,n = match_image_nwd(r['gt_px'], pxy, pcf, nwd_thr, C)
        tp+=t; fp+=f; fn+=n
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)

def load(path):
    d = json.load(open(path))
    return [{'gt_px': r['gt']['xyxy'], 'pred_px': r['pred']['xyxy'],
              'pred_conf': r['pred'].get('confidence', [])} for r in d['records']]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-dir', required=True)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--conf', type=float, default=0.5, help='confidence threshold (fixed while sweeping NWD)')
    ap.add_argument('--nwd-thresholds', type=float, nargs='+', default=[0.3,0.5,0.6,0.7,0.8,0.9])
    ap.add_argument('--conf-grid', type=float, nargs='+', default=[round(x,2) for x in np.arange(0.1,0.91,0.1)])
    ap.add_argument('--nwd-for-conf-sweep', type=float, default=0.3, help='fixed NWD threshold when sweeping confidence')
    ap.add_argument('--c-const', type=float, default=None, help='override C; default = median GT sqrt-area')
    args = ap.parse_args()

    det_dir = os.path.expanduser(args.det_dir)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(
        os.path.dirname(os.path.dirname(det_dir)), 'report_nwd_' + os.path.basename(det_dir))
    os.makedirs(out_dir, exist_ok=True)

    models = {}
    for _f in sorted(glob.glob(os.path.join(det_dir, 'detections_bg*.json'))):
        _pct = os.path.basename(_f).replace('detections_bg', '').replace('.json', '')
        models[f'{int(_pct)}% bg'] = _f
    if not models:
        print(f'WARNING: no detections_bg*.json files found in {det_dir}')
    data = {}
    for name, p in models.items():
        if os.path.exists(p):
            data[name] = load(p)
        else:
            print(f'SKIP {name}: not found ({p})')
    assert data, 'no detections loaded'

    # C: median GT sqrt-area across all loaded models (pooled), unless overridden
    if args.c_const is not None:
        C = args.c_const
    else:
        areas = []
        for recs in data.values():
            for r in recs:
                for g in r['gt_px']:
                    w = g[2]-g[0]; h = g[3]-g[1]
                    areas.append((w*h)**0.5)
        C = float(np.median(areas)) if areas else 32.0
    print(f'Using C = {C:.2f} (median GT sqrt-area of this benchmark)')
    print(f'Fixed confidence threshold: {args.conf}\n')

    rows = []
    print('=== NWD-THRESHOLD SWEEP (replaces IoU sweep; same conf, same detections) ===')
    for name, recs in data.items():
        print(f'-- {name} --')
        for thr in args.nwd_thresholds:
            m = confusion_nwd(recs, args.conf, thr, C)
            rows.append([name, thr, m['precision'], m['recall'], m['f1'], m['tp'], m['fp'], m['fn']])
            print(f'   NWD>={thr:.2f}: P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f} '
                  f'TP/FP/FN={m["tp"]}/{m["fp"]}/{m["fn"]}')

    with open(os.path.join(out_dir, 'nwd_sweep.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model','nwd_threshold','precision','recall','f1','TP','FP','FN'])
        w.writerows([[a, b, round(c,4), round(d,4), round(e,4), g, h, i] for a,b,c,d,e,g,h,i in rows])
    print(f'\nsaved {os.path.join(out_dir, "nwd_sweep.csv")}')

    # ---------------- confidence sweep at fixed NWD threshold ----------------
    print(f'\n=== CONFIDENCE SWEEP (NWD>={args.nwd_for_conf_sweep}) ===')
    conf_rows = []
    for name, recs in data.items():
        print(f'-- {name} --')
        for c in args.conf_grid:
            m = confusion_nwd(recs, c, args.nwd_for_conf_sweep, C)
            conf_rows.append([name, c, m['precision'], m['recall'], m['f1'], m['tp'], m['fp'], m['fn']])
            print(f'   conf {c:.1f}: P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f} '
                  f'TP/FP/FN={m["tp"]}/{m["fp"]}/{m["fn"]}')
    with open(os.path.join(out_dir, 'nwd_conf_sweep.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model','conf','precision','recall','f1','TP','FP','FN'])
        w.writerows([[a, b, round(c,4), round(d,4), round(e,4), g, h, i] for a,b,c,d,e,g,h,i in conf_rows])
    print(f'saved {os.path.join(out_dir, "nwd_conf_sweep.csv")}')

    # ---------------- plots ----------------
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4.5))
    for name, recs in data.items():
        fps = [confusion_nwd(recs, c, args.nwd_for_conf_sweep, C)['fp'] for c in args.conf_grid]
        plt.plot(args.conf_grid, fps, marker='o', label=name)
    plt.xlabel('confidence threshold'); plt.ylabel('false positives (total, 98 img)')
    plt.title(f'False positives vs confidence (NWD>={args.nwd_for_conf_sweep}, C={C:.1f})')
    plt.legend(); plt.grid(alpha=0.3); plt.yscale('log')
    plt.savefig(os.path.join(out_dir, 'nwd_fp_vs_conf.png'), dpi=150, bbox_inches='tight'); plt.close()

    plt.figure(figsize=(6,4.5))
    for name, recs in data.items():
        f1s = [confusion_nwd(recs, c, args.nwd_for_conf_sweep, C)['f1'] for c in args.conf_grid]
        plt.plot(args.conf_grid, f1s, marker='s', label=name)
    plt.xlabel('confidence threshold'); plt.ylabel('F1')
    plt.title(f'F1 vs confidence (NWD>={args.nwd_for_conf_sweep}, C={C:.1f})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'nwd_f1_vs_conf.png'), dpi=150, bbox_inches='tight'); plt.close()

    plt.figure(figsize=(6,4.5))
    for name, recs in data.items():
        P, R = [], []
        for c in args.conf_grid:
            m = confusion_nwd(recs, c, args.nwd_for_conf_sweep, C)
            P.append(m['precision']); R.append(m['recall'])
        plt.plot(R, P, marker='.', label=name)
    plt.xlabel('recall'); plt.ylabel('precision')
    plt.title(f'Precision-Recall (NWD>={args.nwd_for_conf_sweep}, conf-swept, C={C:.1f})')
    plt.xlim(0,1); plt.ylim(0,1.02); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'nwd_pr_curves.png'), dpi=150, bbox_inches='tight'); plt.close()

    print('saved nwd_fp_vs_conf.png, nwd_f1_vs_conf.png, nwd_pr_curves.png')
    print(f'\nC used: {C:.2f}  (report this in the paper alongside the numbers)')

if __name__ == '__main__':
    main()
