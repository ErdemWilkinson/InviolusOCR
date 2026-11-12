# Night 2026-11-12 plan: distillation from a pretrained teacher (written 20:30, before any result)

Owner approved on 2026-11-11 ("önerini kabul ediyorum"), after a three-subagent review
(statistician, tiny-ML, SAR field). Download approved: Keras EfficientNet-B0 ImageNet weights (16.7 MB).

## Question
Does a teacher that brings knowledge from outside our data (ImageNet) help nano tell people
from look-alikes? The 2026-11-11 teachers (width 2, from scratch) learned from the same 706
photos and were no better than v3 size, so they had nothing to teach.

## Teacher (never deployed)
- `train.py` with `VOX_ARCH=effb0`: EfficientNet-B0 (ImageNet) on the tile resized to 320 px,
  strides 8/16/32 merged into the 40x40 head. Data `voxhod_g` (held-out-location split, no NOMAD),
  batch 8, lr 5e-4, 60 min, float, no QAT. Seed 1.
- train.py writes the last-epoch and the best-val-loss weights. **Which one becomes the teacher is
  decided on val only**, by the gate metric below.

## Teacher gate (val only; test is not looked at for this decision)
- Val false alarms per tile at the highest threshold with val recall >= 0.781 (the target used
  since 2026-11-10). Reference: mean of K1L, K2L, K3L (nano recipe, int8) at the same rule.
- **Pass if teacher FA <= 0.70 x reference mean.** If neither checkpoint passes: no student runs;
  the GPU night is used for baseline seeds 4 and 5 only, and the result is "no good teacher".

## Students (if the teacher passes)
- Nano recipe as K1L-K3L (width 0.5, lr 2e-3, 30 min, then 10 min QAT from the last epoch),
  plus the teacher heatmap as a 5th target channel, loss + 1.0 x sum((pred - teacher)^2) / centres.
- Seeds 1-5: D1-D5. Baseline: K1L-K3L (done 2026-11-11, same code path) + K4L, K5L (seeds 4, 5).
- Two runs in parallel on the GPU.

## Student decision rule (new gate, fixed now)
Threshold per run: lowest threshold with val FA per tile <= 0.610 (GA1's val FA at 0.20).
Hierarchical paired bootstrap (`scripts/arms_hier.py`, `ARMS_PAIRED=1`): resample seed pairs,
then test photos / NOMAD frames; 2000 draws.
- **Primary (improvement):** HERIDAL test recall at 0.3 false alarms per tile (curve),
  D - K: lower 95% bound > 0.
- **Guards:** HERIDAL recall at the budget threshold, D - K: lower 95% bound >= -2 points.
  NOMAD false alarms at the budget threshold: point estimate not higher by more than 0.05/tile.
  WiSARD (one clip, descriptive): top-2 frame hit not lower by more than 3 points.
- Size: unchanged by construction (same graph).
- Even if it passes, nothing is published tonight: these students train on the 55% split.
  A pass means "repeat on the full training data", not "ship".

## Also reported (no decision)
- Top-K recall per frame (`scripts/topk_recall.py`) for all runs.
- Teacher's own test numbers (descriptive, after the gate decision).

## Amendment 20:50 (before any student run or teacher result)
- A CPU unit check showed the distillation term normalised per person centre was ~20x the focal
  term (91 vs 4.4 with a dummy teacher), which would drown the detection loss. It is now normalised
  per tile: `KD_WEIGHT * sum((pred - teacher)^2) / tiles_in_batch`, weight 1.0. No other change.

## Teacher gate result 21:30 (val only)
- Reference (K1L/K2L/K3L, int8): val FA at recall >= 0.781: 0.463 / 0.383 / 0.441, mean 0.429; gate <= 0.300.
- T0 last epoch (19): val FA 0.037 at recall 0.786 (ratio 0.09) -> PASS. T0 best val-loss epoch (8): 0.041 -> PASS.
- Teacher = T0 last (lower gate metric). Val recall at the 0.61 FA budget: teacher 0.931 vs nano 0.800-0.815.
- Students paused at the owner's request (he is using the PC); start D1-D5 + K4L, K5L when he says go.

## Known caveats
- 5 seeds is still few for the run-level resample; intervals will be wide.
- The teacher sees the tile at 320 px but only the pixels of the 160 px tile (upsampled).
- No NOMAD in training tonight; NOMAD is a held-out check only.

## Amendment 03:15, 2026-11-12 (students running, no student result exists yet)
- The owner gave GPU time until 10:00. Use it to add seeds, decided now, before any student result:
  **D6-D8 (seeds 6-8, same teacher) and K6-K8 (seeds 6-8, no teacher)**. The decision is taken ONCE,
  on all 8 + 8 runs (D1-D8 vs K1L,K2L,K3L,K4-K8). No interim decision on the first 5 seeds.
- Rule, thresholds and guards unchanged. If a run fails or is not finished by ~09:00, the analysis uses
  the runs that are complete, and the report says which ones are missing.
