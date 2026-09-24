# Lessons

What 1,362 scans, 17 submissions and several hundred cross-validated arms taught us. Each line is a
measured result, not an opinion.

## Validation

| lesson | evidence |
|---|---|
| **Group the folds by acquisition site.** | Scanner metadata alone predicts the label at AUROC 0.726 under random folds and 0.533 under grouped folds. Random CV rewards recognising the hospital. |
| **Cross-fit every stage, not just the last.** | Nesting only the second stage of a stacked model leaked 0.046 log loss and produced a fake +0.0096 candidate. |
| **Price selection before believing a gain.** | The best of 297 searched arms keeps only 0.24 of its apparent edge. After that haircut, none of our searched candidates had a real one. |
| **Leave-one-group-out cannot validate a per-group correction.** | A held-out group cannot estimate its own offset. Cross-fit within the group instead. |
| **A single fold is not a screen.** | Across 316 arms, fold 0 of seed 0 ran 0.028 log loss harder than the others. |
| **Control against the same pipeline, not a remembered number.** | A stacking control scored 0.2516 against a raw 0.2448, so a "+0.0072 gain" was really +0.0004. |

## Modelling

| lesson | evidence |
|---|---|
| **Normalise each scan by its own reference level.** | It removes a 16,000-fold between-centre intensity range, using no cohort statistic. |
| **A weak, different member beats a strong, similar one.** | The SBR feature head scores 0.4404 alone against ~0.27 for the CNNs, yet removing it costs 0.0055. No extra CNN ever added that much. |
| **Decorrelation is necessary, not sufficient, and it is one seat only.** | A second natural-image backbone sat 0.80 correlated with the first and added +0.0003. |
| **TTA pays only on transforms the model never saw.** | The siamese net is left-right invariant by construction, so mirror TTA is an exact no-op for it; superior-inferior views are worth +0.029. |
| **Label smoothing hurt, monotonically.** | Turning it off was worth +0.0044 in the blend, on all three seeds. |
| **Pool training draws, never pick the luckier one.** | 011 ships both count-noise draws (30 checkpoints); shipping only the better draw would be selecting on a coin flip. |

## The leaderboard

| lesson | evidence |
|---|---|
| **Past CV 0.247, a better CV score did not transfer.** | Nine challengers were flown against 008a and all nine lost publicly (sign test p = 0.002). Our best CV ever (0.2321) posted our worst public score (0.2815). |
| **The public split is small.** | Public scores rest on about 145 scans; the standard error of a difference between two models is about 0.008 to 0.018 log loss. |
| **Freeze the reading before the score lands.** | Every flight had its interpretation bands committed beforehand, so no result could be re-read to taste. |
| **Expect a shake-up.** | 008a went from 0.2653 public to 0.3054 private. |

## Engineering

| lesson | evidence |
|---|---|
| **MedicalNet training is not bit-reproducible.** | PyTorch has no deterministic `max_pool3d_with_indices_backward_cuda`; retrains move that family by ~0.002 and the blend by ~0.0003. |
| **A float16 cache always differs from float32 by one ULP.** | That difference looks exactly like a preprocessing bug and is not one. |
| **Detect garbage scans, do not trust them.** | One NaN voxel sends the reference level to 1.0 and saturates the cube, flipping a normal scan from p = 0.08 to p = 0.67. Inference falls back to the training prevalence instead. |
| **Test the shipped archive, not the staging tree.** | Every pack ran the test suite, a local inference and a container run before upload. |
