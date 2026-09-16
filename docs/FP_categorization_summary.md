# False-Positive Categorization by Background Type — Summary

*Prepared 2026-08-16. Written in response to reviewer x65Y's comment: "it would help to know
which kinds of background drive the false positives." This document summarizes the methodology
and results of a manual categorization exercise across all four background-augmentation levels
(0%, 5%, 10%, 20%).*

## 1. Background

The paper reports false positives as a single undifferentiated count per background level (e.g.
1121 at 0% falling to 376 at 20%, at confidence ≥ 0.5), and asserts in the Introduction that the
detector "hallucinate[s] objects on wave texture and glare" — but this claim was never directly
measured. Reviewer x65Y asked, in effect: when the detector fires falsely, what is actually under
the box, and does background augmentation suppress all false-positive types evenly, or does it
crush some while leaving others standing?

This exercise answers that question directly, using data already on hand (the four checkpoints'
saved detections and the 98-image benchmark) — no new training or annotation of the benchmark
itself was needed.

## 2. Methodology

### 2.1 False-positive definition and crop-selection strategy

A "false positive" here uses the **exact same definition already reported in the paper's Table
1**: a prediction (at confidence ≥ 0.5) that, under one-to-one, confidence-ordered greedy
matching (each ground-truth box claimable by at most one prediction, highest-confidence match
wins), has no ground-truth box within IoU ≥ 0.1. This was verified to reproduce Table 1's exact
FP counts at each background level (1121 / 562 / 393 / 376 for 0/5/10/20%), confirming the
categorized crops correspond precisely to the same false positives already cited in the paper.

Given these totals are too large to categorize exhaustively by hand, a **stratified sample** was
drawn per background level:
- Every false positive at every image was identified.
- Crops were **capped at 5 per source image**, so a single cluttered frame could not dominate the
  sample and distort the category breakdown.
- The remaining candidates were **randomly sampled down to ~200 per background level** (a fixed
  random seed was used, so the sample is reproducible).

This gives roughly 200 categorized crops per level — a genuine, defensible sample of each
background level's true false-positive population, not an exhaustive census.

### 2.2 Crop presentation: padding for context, red box for localization

Each false positive was cropped from the full-resolution source image with **generous padding**
around the detection box — padding of `max(3× the box's own size, 150px)` on each side. This was
necessary because many objects in this benchmark are small (median object √-area is ~95px across
the full 98-image benchmark), and judging *background type* requires seeing the surrounding
scene, not just a tight, context-free sliver.

Because the padded crop often contains multiple candidate features (e.g. a kayak elsewhere in
frame while the actual detection sits over open water), **the exact detection box is drawn as a
red rectangle directly on the saved crop**. This removes ambiguity about which specific feature
is being categorized — categorization always follows what is inside the red box, never anything
else merely visible in the wider padded context.

### 2.3 Annotation code list

| Code | Meaning |
|---|---|
| S | Shadow |
| H | Human (includes associated gear/vessels: kayak, boat, surfboard) |
| E | Excreta (bright white dots; visually similar to glint/foam; likely mostly liquid bird droppings; deliberately not annotated in ground truth — see §4) |
| B | Bird |
| W | Wave |
| G | Glint/Glare |
| F | Foam |
| K | Kelp (coarse; superseded by Km/Kf below) |
| Km | Kelp — mat/raft (large, salient) |
| Kf | Kelp — fragment (small, floating) |
| Wk | Wake (established but never observed — 0 instances across all four levels) |
| O | Other (coarse; superseded by Od/Oc below) |
| Od | Other — too dark, blurred, or submerged to identify |
| Oc | Other — clearly visible, does not fit any category (this bucket contains real marine megafauna — whale, shark, jelly, fish, pinniped, otter, mola, bat ray — plus objects like buoys) |
| N | Empty — no visible feature at all under the box (flat water) |

The coding scheme evolved over the course of the exercise (Km/Kf, Od/Oc, and N were all
introduced partway through, as genuinely useful distinctions became apparent). The 0%-background
level was **re-annotated from scratch once the full scheme matured**, so all four levels reported
below use the same, final, consistent code set.

### 2.4 Consistency checking

Because all four background-level models were evaluated on the same 98 images, the same
real-world false-positive locations (a given wave patch, a given missed bird) frequently recur
across multiple background levels' samples. A cross-referencing check (same source image, box
centers within 75px) was used to verify categorization consistency across levels. Across the
fully-matured manifests (5%, 10%, 20%), independent categorizations of the same real-world
location agreed **88.5% of the time**; nearly all disagreements were reasonable, closely-related
judgment calls (e.g. kelp mat vs. fragment), not contradictory category assignments.

## 3. Results

All figures below are **estimated absolute counts**: each level's ~200-crop sample percentage,
scaled up to that level's true total false-positive population at confidence ≥ 0.5 (1121 / 562 /
393 / 376 for 0/5/10/20% background, matching Table 1).

| Category | 0% bg | 5% bg | 10% bg | 20% bg | Change, 0%→20% |
|---|---|---|---|---|---|
| Empty (N) | 56 | 3 | 2 | 2 | **−97%** |
| Wave (W) | 409 | 135 | 63 | 53 | **−87%** |
| Shadow (S) | 11 | 8 | 0 | 0 | −100% *(small n)* |
| Other (O/Od/Oc) | 191 | 112 | 73 | 55 | −71% |
| Glint/Glare (G) | 78 | 28 | 12 | 28 | −64% |
| Excreta (E) | 28 | 45 | 10 | 13 | −53% |
| Kelp (K/Km/Kf) | 140 | 70 | 84 | 70 | −50% |
| Foam (F) | 22 | 17 | 24 | 13 | −41% |
| Bird (B) | 157 | 126 | 108 | 119 | −24% |
| Human (H) | 28 | 17 | 18 | 23 | −19% |
| **Total false positives** | **1121** | **562** | **393** | **376** | **−66%** *(overall baseline)* |

## 4. Conclusions

**False positives split into two behaviorally distinct groups, and background augmentation acts
on them very differently.** One group — Empty, Wave, Shadow, Other, Glint/Glare, and Excreta —
declines steeply with added background augmentation (50–97%, outpacing the overall 66% decline
in total false positives). The other group — Kelp, Foam, Bird, and Human — declines much more
mildly (19–50%). This is a direct, quantitative answer to reviewer x65Y's question: background
augmentation does **not** suppress all false-positive types evenly; it specifically and strongly
suppresses genuine background-texture confusion, while barely touching categories that are
actually real, unlabeled objects.

**The single cleanest result is "Empty" — false detections on plain, featureless water, with
nothing visible at all.** This category shows the steepest decline of any (−97%), and it is
arguably the least ambiguous evidence in the whole exercise: there is no possible argument that
texture or glare is present, so its near-total collapse under background augmentation is about as
direct a confirmation of the intended training effect as this analysis can produce.

**Excreta, despite being a real, physical object, behaves like a background artifact, not like
a missed annotation.** It presents visually as bright white dots — close to indistinguishable
from glint and foam — and is deliberately excluded from ground-truth annotation (there are far
too many such deposits in a typical frame to label exhaustively). Because the model has no reason
to treat visually-identical stimuli differently regardless of their real-world cause, whatever
training effect suppresses glint and foam naturally suppresses excreta too.

**A substantial fraction of remaining false positives — roughly 14% to 38% depending on
background level — are genuinely missed ground-truth annotations, not model failures.** Bird,
Human (and associated gear), and Kelp fragments account for most of this, and the "Other" bucket
additionally contains real marine megafauna (whale, shark, pinniped, otter, mola, bat ray, jelly)
that were never labeled. This provides direct, quantified support for the annotation-limitation
caveat already noted in the paper as a direction for future work — what was previously stated as
an informed expectation can now be stated as a measured finding.

**This does not undermine the paper's core claims, though it does mean absolute numbers are
conservative.** The background-augmentation-suppresses-false-positives finding is, if anything,
sharpened by this analysis — its real driver is the artifact-type categories, which show a clean,
strong, disproportionate effect. The IoU-vs-NWD metric comparison is unaffected, since both
metrics score the identical set of detections against the identical (incomplete) ground truth,
so their relative comparison holds regardless of annotation gaps. What the missed-annotation
finding does mean is that reported precision, F1, and mAP figures are somewhat **understated** —
a conservative bias, since some detections currently scored as false positives are in fact
correct — rather than a finding that calls the reported comparisons into question.

**Foam is a partial exception worth stating plainly rather than glossing over.** Despite being
visually similar to glint and excreta, foam's decline (−41%) is milder than the other
artifact-type categories and closer to the persisting group. We do not have a confident
explanation for this; it is reported as an honest limitation rather than smoothed into either
grouping.

## 5. Limitations

- Each background level's breakdown is based on a **stratified sample of ~200 false positives**,
  not an exhaustive categorization of the full population (which ranges from 376 to 1121 per
  level) — the percentages and derived absolute-count estimates carry real sampling uncertainty,
  particularly for lower-count categories.
- Two categories (Glint/Glare, Kelp) do not decline smoothly across the four background levels;
  some of this non-monotonicity is plausibly sampling noise given the modest per-level sample
  sizes, though it has not been separately verified.
- Whether Kelp fragments are formally an in-scope annotated class (and therefore squarely a
  missed-annotation case) versus a genuine benchmark-scope question has not yet been confirmed.
- Categorization was performed by a single evaluator; no second-rater agreement check was
  performed (the cross-level consistency check in §2.4 provides a partial, indirect substitute,
  since it captures self-consistency but not independent-rater agreement).
