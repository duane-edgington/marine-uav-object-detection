#!/usr/bin/env python3
"""
report_local.py — background-fraction comparison report (CPU), broad sweeps.

Reads detections_bg<N>.json (from rf_detr_multimodel_bg_compare_LOCAL.py) and emits:
  - conf_sweep.csv        : P/R/F1/TP/FP/FN per model at conf 0.1..0.9 (IoU fixed)
  - iou_sweep.csv         : P/R/F1 per model across IoU 0.3/0.5/0.75/0.9 (conf fixed)
  - model_summary.csv     : mAP50, mAP50-95 per model (compact reference)
  - fp_vs_conf_iou<NN>.png, f1_vs_conf_iou<NN>.png, pr_curves_iou<NN>.png : one set per IoU in
                            FP_F1_PR_IOU_LIST (default 0.3 and 0.5); 0.3 is also copied to the
                            original fp_vs_conf.png / f1_vs_conf.png / pr_curves.png names.
  - confusion_matrices_iou<NN>.png: grid of TP/FP/FN confusion matrices, models x confidence
                            thresholds (object-detection has no true-negative; N/A cell)
  - pr_curves.png         : precision-recall (conf-swept) per model
  - f1_vs_conf.png        : F1 vs confidence per model

Run (defaults to the config below):
  python report_local.py
Or override the run folder without editing this file:
  python report_local.py --det-dir ~/rfdetr-work/model_compare/onnx_slice384_v1
  python report_local.py --det-dir ~/rfdetr-work/model_compare/onnx_slice384_v1 --out-dir ~/rfdetr-work/report_onnx_slice384_v1
If --out-dir is omitted, it defaults to '<parent of det-dir's parent>/report_<det-dir folder name>'.
"""
import os, json, argparse, glob
import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument('--det-dir', default=None, help='folder with detections_bg*.json (overrides DET_DIR below)')
_ap.add_argument('--out-dir', default=None, help='folder to write report outputs (overrides OUTPUT_DIR below)')
_args, _ = _ap.parse_known_args()

# ---------------- CONFIG ----------------
_DEFAULT_DET_DIR = os.path.expanduser('~/rfdetr-work/model_compare/sarah_clean_v3')
DET_DIR = os.path.expanduser(_args.det_dir) if _args.det_dir else _DEFAULT_DET_DIR
if _args.out_dir:
    OUTPUT_DIR = os.path.expanduser(_args.out_dir)
elif _args.det_dir:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(DET_DIR)), 'report_' + os.path.basename(DET_DIR))
else:
    OUTPUT_DIR = os.path.expanduser('~/rfdetr-work/report_v3')
MODELS = {}
for _f in sorted(glob.glob(os.path.join(DET_DIR, 'detections_bg*.json'))):
    _pct = os.path.basename(_f).replace('detections_bg', '').replace('.json', '')
    MODELS[f'{int(_pct)}% bg'] = _f
if not MODELS:
    print(f'WARNING: no detections_bg*.json files found in {DET_DIR}')
CONF_GRID = [round(x,2) for x in np.arange(0.1, 0.91, 0.1)]     # 0.1 .. 0.9
IOU_GRID  = [0.3, 0.5, 0.75, 0.9]
CONF_FOR_IOU_SWEEP = 0.5     # fixed conf when sweeping IoU
FP_F1_PR_IOU_LIST = [0.3, 0.5]  # generate fp/F1/PR figures at EACH of these IoUs (0.3 recommended
                                 # as the primary figure for this benchmark's object scale --
                                 # IoU 0.5 collapses to near-zero TP and is included for reference)
IOU_FOR_CONF_SWEEP = FP_F1_PR_IOU_LIST[0]   # kept for the printed confidence-sweep table above
CM_CONF_LIST = [0.1, 0.3, 0.5, 0.7]   # confidence thresholds shown in each confusion-matrix grid
CM_IOU_LIST = [0.1, 0.3, 0.5]         # one grid PNG per IoU level. 0.1 = "detected nearby" (the
                                       # real detection rate, ~43% of GT per prior analysis); 0.3 =
                                       # loose localization (signal drops off hard, ~2%); 0.5 =
                                       # standard threshold (near-total collapse). Skip 0.7+:
                                       # identical to 0.5 once TP hits 0 there -- no new info.
IOU_LIST_MAP = [round(x,2) for x in np.arange(0.5,1.0,0.05)]    # for mAP50-95
os.makedirs(OUTPUT_DIR, exist_ok=True)
import matplotlib; matplotlib.use('Agg')

# ---------------- metrics engine (tested) ----------------
def iou_matrix(gt, pred):
    if len(gt)==0 or len(pred)==0: return np.zeros((len(gt),len(pred)))
    gt=np.asarray(gt,float); pred=np.asarray(pred,float)
    x1=np.maximum(gt[:,None,0],pred[None,:,0]); y1=np.maximum(gt[:,None,1],pred[None,:,1])
    x2=np.minimum(gt[:,None,2],pred[None,:,2]); y2=np.minimum(gt[:,None,3],pred[None,:,3])
    iw=np.clip(x2-x1,0,None); ih=np.clip(y2-y1,0,None); inter=iw*ih
    ag=((gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1]))[:,None]; ap=((pred[:,2]-pred[:,0])*(pred[:,3]-pred[:,1]))[None,:]
    return inter/np.clip(ag+ap-inter,1e-9,None)

def match_image(gt, pred_xyxy, pred_conf, iou_thr):
    order=np.argsort(-np.asarray(pred_conf)) if len(pred_conf) else np.array([],int)
    pred=np.asarray(pred_xyxy,float)[order] if len(pred_xyxy) else np.zeros((0,4))
    conf=np.asarray(pred_conf,float)[order] if len(pred_conf) else np.zeros((0,))
    iou=iou_matrix(gt,pred); matched=set(); tp=np.zeros(len(pred),bool)
    for j in range(len(pred)):
        if iou.shape[0]==0: break
        col=iou[:,j].copy()
        for g in matched: col[g]=-1
        g=int(np.argmax(col)) if col.size else -1
        if col.size and col[g]>=iou_thr: tp[j]=True; matched.add(g)
    return conf,tp,len(gt)

def confusion(records, conf_thr, iou_thr):
    tp=fp=fn=0
    for r in records:
        keep=[i for i,c in enumerate(r['pred_conf']) if c>=conf_thr]
        pxy=[r['pred_px'][i] for i in keep]; pcf=[r['pred_conf'][i] for i in keep]
        _,t,g=match_image(r['gt_px'],pxy,pcf,iou_thr)
        tpn=int(t.sum()); tp+=tpn; fp+=len(t)-tpn; fn+=g-tpn
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
    return dict(tp=tp,fp=fp,fn=fn,precision=prec,recall=rec,f1=f1)

def ap_coco101(conf_all,tp_all,n_gt):
    if n_gt==0: return float('nan')
    if len(conf_all)==0: return 0.0
    o=np.argsort(-conf_all); tp=tp_all[o].astype(float); fp=1-tp
    ctp=np.cumsum(tp); cfp=np.cumsum(fp); rec=ctp/n_gt; prec=ctp/np.clip(ctp+cfp,1e-9,None)
    rc=np.linspace(0,1,101); p=np.zeros(101)
    for i,rr in enumerate(rc):
        m=rec>=rr; p[i]=prec[m].max() if m.any() else 0.0
    return p.mean()

def mean_ap(records, iou_list):
    aps={}
    for thr in iou_list:
        C=[];T=[];ng=0
        for r in records:
            c,t,g=match_image(r['gt_px'],r['pred_px'],r['pred_conf'],thr); C.append(c);T.append(t);ng+=g
        C=np.concatenate(C) if C else np.zeros(0); T=np.concatenate(T) if T else np.zeros(0,bool)
        aps[thr]=ap_coco101(C,T,ng)
    return aps

# ---------------- load ----------------
def load(path):
    d=json.load(open(path))
    return [{'gt_px':r['gt']['xyxy'],'pred_px':r['pred']['xyxy'],'pred_conf':r['pred'].get('confidence',[])} for r in d['records']]

data={}
for name,p in MODELS.items():
    if not os.path.exists(p): print('SKIP',name,'(missing)'); continue
    data[name]=load(p)
assert data, 'no detections loaded'
for name,recs in data.items():
    ng=sum(len(r['gt_px']) for r in recs)
    print(f'{name}: {len(recs)} imgs, {ng} GT boxes')

import csv
# ---------------- confidence sweep (IoU fixed) ----------------
print(f'\n=== CONFIDENCE SWEEP  (IoU={IOU_FOR_CONF_SWEEP}) ===')
rows=[]
for name,recs in data.items():
    for c in CONF_GRID:
        m=confusion(recs,c,IOU_FOR_CONF_SWEEP)
        rows.append([name,c,m['precision'],m['recall'],m['f1'],m['tp'],m['fp'],m['fn']])
    print(f'-- {name} --')
    for c in CONF_GRID:
        m=confusion(recs,c,IOU_FOR_CONF_SWEEP)
        print(f'   conf {c:.1f}: P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f}  TP/FP/FN={m["tp"]}/{m["fp"]}/{m["fn"]}')
with open(os.path.join(OUTPUT_DIR,'conf_sweep.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['model','conf','precision','recall','f1','TP','FP','FN']); w.writerows([[a,b,round(c,4),round(d,4),round(e,4),g,h,i] for a,b,c,d,e,g,h,i in rows])

# ---------------- IoU sweep (conf fixed) ----------------
print(f'\n=== IoU SWEEP  (conf={CONF_FOR_IOU_SWEEP}) ===')
irows=[]
for name,recs in data.items():
    print(f'-- {name} --')
    for iou in IOU_GRID:
        m=confusion(recs,CONF_FOR_IOU_SWEEP,iou)
        irows.append([name,iou,m['precision'],m['recall'],m['f1'],m['tp'],m['fp'],m['fn']])
        print(f'   IoU {iou:.2f}: P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f}  TP/FP/FN={m["tp"]}/{m["fp"]}/{m["fn"]}')
with open(os.path.join(OUTPUT_DIR,'iou_sweep.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['model','iou','precision','recall','f1','TP','FP','FN']); w.writerows([[a,b,round(c,4),round(d,4),round(e,4),g,h,i] for a,b,c,d,e,g,h,i in irows])

# ---------------- mAP summary ----------------
print('\n=== mAP SUMMARY (compact reference) ===')
srows=[]
for name,recs in data.items():
    aps=mean_ap(recs,IOU_LIST_MAP); m50=aps[0.5]; m5095=np.nanmean(list(aps.values()))
    srows.append([name,round(m50,4),round(m5095,4)])
    print(f'   {name}: mAP50={m50:.4f}  mAP50-95={m5095:.4f}')
with open(os.path.join(OUTPUT_DIR,'model_summary.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['model','mAP50','mAP50_95']); w.writerows(srows)

# ---------------- plots ----------------
import matplotlib.pyplot as plt

for _iou in FP_F1_PR_IOU_LIST:
    tag = str(_iou).replace('.', '')  # e.g. 0.3 -> '03', 0.5 -> '05' (matches confusion-matrix naming)

    # FP vs confidence
    plt.figure(figsize=(6,4.5))
    for name,recs in data.items():
        fps=[confusion(recs,c,_iou)['fp'] for c in CONF_GRID]
        plt.plot(CONF_GRID,fps,marker='o',label=name)
    plt.xlabel('confidence threshold'); plt.ylabel('false positives (total, 98 img)')
    plt.title(f'False positives vs confidence (IoU {_iou})'); plt.legend(); plt.grid(alpha=0.3); plt.yscale('log')
    plt.savefig(os.path.join(OUTPUT_DIR,f'fp_vs_conf_iou{tag}.png'),dpi=150,bbox_inches='tight'); plt.close()

    # F1 vs confidence
    plt.figure(figsize=(6,4.5))
    for name,recs in data.items():
        f1s=[confusion(recs,c,_iou)['f1'] for c in CONF_GRID]
        plt.plot(CONF_GRID,f1s,marker='s',label=name)
    plt.xlabel('confidence threshold'); plt.ylabel('F1'); plt.title(f'F1 vs confidence (IoU {_iou})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR,f'f1_vs_conf_iou{tag}.png'),dpi=150,bbox_inches='tight'); plt.close()

    # PR curves (conf-swept, IoU fixed)
    plt.figure(figsize=(6,4.5))
    for name,recs in data.items():
        P=[];R=[]
        for c in CONF_GRID:
            m=confusion(recs,c,_iou); P.append(m['precision']); R.append(m['recall'])
        plt.plot(R,P,marker='.',label=name)
    plt.xlabel('recall'); plt.ylabel('precision'); plt.title(f'Precision-Recall (IoU {_iou}, conf-swept)')
    plt.xlim(0,1); plt.ylim(0,1.02); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR,f'pr_curves_iou{tag}.png'),dpi=150,bbox_inches='tight'); plt.close()

    print(f'saved fp_vs_conf_iou{tag}.png, f1_vs_conf_iou{tag}.png, pr_curves_iou{tag}.png')

# Back-compat: also write the original filenames (no IoU suffix) at the FIRST list entry
# (0.3, the recommended primary IoU for this benchmark) so anything referencing the old
# names still works.
import shutil as _shutil
_primary_tag = str(FP_F1_PR_IOU_LIST[0]).replace('.', '')
for _base in ('fp_vs_conf', 'f1_vs_conf', 'pr_curves'):
    _src = os.path.join(OUTPUT_DIR, f'{_base}_iou{_primary_tag}.png')
    _dst = os.path.join(OUTPUT_DIR, f'{_base}.png')
    if os.path.exists(_src):
        _shutil.copyfile(_src, _dst)

# Confusion matrices: one grid PNG per IoU level in CM_IOU_LIST; within each grid,
# rows = models, cols = confidence thresholds. Object-detection has no well-defined true-negative
# (no finite set of "non-object" boxes), so that cell is left as N/A rather than faking a number.
for cm_iou in CM_IOU_LIST:
    n_rows, n_cols = len(data), len(CM_CONF_LIST)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1*n_cols, 3.0*n_rows), squeeze=False)
    for i,(name,recs) in enumerate(data.items()):
        for j,c in enumerate(CM_CONF_LIST):
            m = confusion(recs, c, cm_iou)
            ax = axes[i][j]
            grid = np.array([[m['tp'], m['fn']],
                              [m['fp'], np.nan]])
            disp = np.where(np.isnan(grid), 0, grid)
            im = ax.imshow(disp, cmap='Blues', vmin=0)
            for (r,cc), val in np.ndenumerate(grid):
                txt = 'N/A' if np.isnan(val) else f'{int(val)}'
                ax.text(cc, r, txt, ha='center', va='center',
                        color='white' if (not np.isnan(val) and val > disp.max()*0.5) else 'black',
                        fontsize=11, fontweight='bold')
            ax.set_xticks([0,1]); ax.set_xticklabels(['Pred: Object','Pred: None'], fontsize=8)
            ax.set_yticks([0,1]); ax.set_yticklabels(['Actual: Object','Actual: None'], fontsize=8)
            if i==0: ax.set_title(f'conf={c}', fontsize=10)
            if j==0: ax.set_ylabel(name, fontsize=10, fontweight='bold')
            ax.tick_params(length=0)
    fig.suptitle(f'Confusion matrices (IoU={cm_iou}) — TP/FN/FP shown; TN undefined for detection',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fname = f'confusion_matrices_iou{str(cm_iou).replace(".","")}.png'
    plt.savefig(os.path.join(OUTPUT_DIR,fname),dpi=150,bbox_inches='tight'); plt.close()
    print(f'saved {fname}')

print('\nAll outputs in', OUTPUT_DIR)
