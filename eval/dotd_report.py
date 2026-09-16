#!/usr/bin/env python3
"""
dotd_report.py — lightweight spot-check: re-evaluate EXISTING detections under Dot Distance
(DotD; Xu, Wang, Yang, Yu, CVPRW 2021, "Dot Distance for Tiny Object Detection in Aerial Images"),
a pure center-distance similarity metric for tiny objects, as a second independent check
alongside NWD (nwd_report.py). NO NEW INFERENCE NEEDED.

DotD formula:
    D = sqrt((cxa-cxb)^2 + (cya-cyb)^2)                     # center Euclidean distance
    S = sqrt( sum(w_i * h_i) / N )                          # RMS avg object area, this benchmark
    DotD = exp(-D / S)

Unlike NWD (which also encodes box width/height via the Gaussian model), DotD depends ONLY on
center position -- simpler, more intuitive, but ignores box shape entirely. Purpose here: confirm
whether the peak-F1-saturation pattern found under NWD (Section 3.5) also holds under DotD. If it
does, that's a second, independent confirmation using a genuinely different (simpler) metric.

Run:
  python dotd_report.py --det-dir ~/rfdetr-work/model_compare/native_v1
"""
import os, json, argparse, csv
import numpy as np

def xyxy_to_cxcy(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2, (y1+y2)/2)

def dotd(a_cxcy, b_cxcy, S):
    dx = a_cxcy[0]-b_cxcy[0]; dy = a_cxcy[1]-b_cxcy[1]
    D = (dx*dx + dy*dy) ** 0.5
    return float(np.exp(-D/S))

def match_image_dotd(gt_xyxy, pred_xyxy, pred_conf, dotd_thr, S):
    order = np.argsort(-np.asarray(pred_conf)) if len(pred_conf) else np.array([], int)
    preds = [pred_xyxy[i] for i in order] if len(pred_xyxy) else []
    gts_c = [xyxy_to_cxcy(g) for g in gt_xyxy]
    preds_c = [xyxy_to_cxcy(p) for p in preds]
    matched = set()
    tp = [False]*len(preds_c)
    for j, p in enumerate(preds_c):
        best_g, best_s = -1, -1.0
        for gi, g in enumerate(gts_c):
            if gi in matched: continue
            s = dotd(g, p, S)
            if s > best_s: best_s, best_g = s, gi
        if best_g >= 0 and best_s >= dotd_thr:
            tp[j] = True; matched.add(best_g)
    return sum(tp), len(preds_c)-sum(tp), len(gts_c)-len(matched)

def confusion(records, conf_thr, dotd_thr, S):
    tp=fp=fn=0
    for r in records:
        keep = [i for i,c in enumerate(r['pred_conf']) if c >= conf_thr]
        pxy = [r['pred_px'][i] for i in keep]; pcf = [r['pred_conf'][i] for i in keep]
        t,f,n = match_image_dotd(r['gt_px'], pxy, pcf, dotd_thr, S)
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
    ap.add_argument('--conf-grid', type=float, nargs='+', default=[round(x,2) for x in np.arange(0.1,0.91,0.1)])
    ap.add_argument('--dotd-thresholds', type=float, nargs='+', default=[0.3,0.5,0.6,0.7,0.8,0.9])
    args = ap.parse_args()

    det_dir = os.path.expanduser(args.det_dir)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(
        os.path.dirname(os.path.dirname(det_dir)), 'report_dotd_' + os.path.basename(det_dir))
    os.makedirs(out_dir, exist_ok=True)

    import glob
    data = {}
    for p in sorted(glob.glob(os.path.join(det_dir, 'detections_bg*.json'))):
        pct = os.path.basename(p).replace('detections_bg', '').replace('.json', '')
        name = f'{int(pct)}% bg'
        data[name] = load(p)
    assert data, f'no detections_bg*.json files found in {det_dir}'

    # S = sqrt(mean box area) across all loaded models' GT (identical GT across models, pooled once)
    areas = []
    for recs in data.values():
        for r in recs:
            for g in r['gt_px']:
                w = g[2]-g[0]; h = g[3]-g[1]; areas.append(w*h)
        break  # GT identical across models; only need one pass
    S = float(np.sqrt(np.mean(areas))) if areas else 32.0
    print(f'Using S = {S:.2f} (RMS average GT object size, this benchmark)\n')

    print('=== PEAK F1 ACROSS CONFIDENCE, PER DotD THRESHOLD ===')
    rows = []
    peak_rows = []
    for name, recs in data.items():
        best_overall = (-1, None, None)
        for thr in args.dotd_thresholds:
            f1s = []
            for c in args.conf_grid:
                m = confusion(recs, c, thr, S)
                rows.append([name, thr, c, m['precision'], m['recall'], m['f1'], m['tp'], m['fp'], m['fn']])
                f1s.append((m['f1'], c, m))
            f1s.sort(reverse=True)
            best_f1, best_conf, best_m = f1s[0]
            if best_f1 > best_overall[0]:
                best_overall = (best_f1, thr, (best_conf, best_m))
        peak_f1, peak_thr, (peak_conf, peak_m) = best_overall
        peak_rows.append([name, peak_thr, peak_conf, peak_f1, peak_m['precision'], peak_m['recall'], peak_m['tp'], peak_m['fp'], peak_m['fn']])
        print(f'  {name:8s}: peak F1={peak_f1:.3f} at DotD>={peak_thr}, conf={peak_conf:.1f} '
              f'(P={peak_m["precision"]:.3f} R={peak_m["recall"]:.3f} TP/FP/FN={peak_m["tp"]}/{peak_m["fp"]}/{peak_m["fn"]})')

    with open(os.path.join(out_dir, 'dotd_full_sweep.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model','dotd_threshold','conf','precision','recall','f1','TP','FP','FN'])
        w.writerows([[a,b,c,round(d,4),round(e,4),round(g,4),h,i,j] for a,b,c,d,e,g,h,i,j in rows])
    with open(os.path.join(out_dir, 'dotd_peak_f1.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model','best_dotd_thr','best_conf','peak_f1','precision','recall','TP','FP','FN'])
        w.writerows(peak_rows)
    print(f'\nsaved {out_dir}/dotd_full_sweep.csv and dotd_peak_f1.csv')
    print(f'S used: {S:.2f}  (report this alongside the numbers)')

if __name__ == '__main__':
    main()
