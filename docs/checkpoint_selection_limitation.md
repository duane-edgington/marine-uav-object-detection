# Known limitation: checkpoint selection vs. deployment-optimal performance

**Status: open, not yet resolved. Documented here so it is visible to readers and reproducers,
not because we consider it disqualifying.**

## The finding

During this project, we found — independently, in both RF-DETR and YOLO26, across multiple
training runs — that the checkpoint a training framework selects as "best" (based on
training-time validation, evaluated on held-out *crops* from the training distribution) is not
reliably the checkpoint that performs best under our actual evaluation methodology (full-image,
tiled inference, scored under NWD/DotD/IoU).

**RF-DETR**: across two independently-trained runs on our 5%-background condition, several
early-training checkpoints (as early as epoch 2, and again around epochs 8–11 in a separate run)
outperformed the checkpoint selected by the framework's own early-stopping criterion (typically
around epoch 20) on our tiled-inference benchmark — despite that selected checkpoint having the
best training-time validation score.

**YOLO26**: our 300-epoch run's framework-selected "best" checkpoint scored notably below the
run's true peak (found around epoch 210) under the same tiled-inference evaluation — performance
climbed for roughly two-thirds of training, then declined over the remaining epochs, and the
final/best-selected checkpoint sat well below the true peak.

## Why this happens (our current best understanding)

Training-time validation is computed on crops drawn from the training-crop distribution (e.g.
576×576 or 640×640 tiles). Our actual evaluation uses full-resolution images sliced into tiles
by an independent inference-time pipeline (SAHI-style), which is a meaningfully different
distribution — different scale statistics, different edge/tiling effects, different background
composition per tile. A checkpoint that has begun to specialize to the training-crop distribution
specifically (rather than to object detection generally) can show improving training-time
validation scores while its performance under the actual deployment-style evaluation stagnates
or declines.

## What we have NOT yet done

- A systematic sweep of intermediate checkpoints across **every** background-augmentation level
  (we have only checked this at one condition per architecture so far).
- A principled replacement for "pick whatever the framework calls best" — e.g., periodically
  evaluating candidate checkpoints against a held-out tiled-inference validation set (matching
  deployment conditions) during training, rather than relying on training-crop validation alone.
- Confirmation of whether this effect is specific to NWD/DotD-style evaluation, or would also be
  visible under standard COCO-style mAP evaluated the same (tiled) way — we suspect the latter,
  since the effect first appeared under mAP inspection before we examined NWD/DotD, but we have
  not run a dedicated comparison.

## What this means for the results in this repository

**All reported model checkpoints in this release are the framework-selected ones** (i.e., the
same checkpoints most practitioners would arrive at by default), not hand-picked
best-under-our-own-evaluation checkpoints. We consider this the more honest and reproducible
choice — reporting a cherry-picked checkpoint chosen after the fact, using the same test data
being reported on, would not be defensible. **The reported numbers may therefore understate the
achievable performance of these training runs**, in a way that varies by architecture and
background level.

We plan to investigate this further — a natural next step is repeating training with periodic
checkpointing evaluated directly against a tiled-inference validation split, for every
background-augmentation condition, and reporting the resulting checkpoint-selection methodology
explicitly. We are documenting it now, at the time of this release, rather than waiting for that
work to be complete, in the interest of transparency about a real, load-bearing limitation of the
current results.
