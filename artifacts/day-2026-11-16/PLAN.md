# 2026-11-16 plan: SARD plus karst hard negatives for nano (written 17:55, before any data or result)

Owner: "aklında bir iş varsa şimdi yapabilirsin" (2026-11-16), after the stop rule discussed on 2026-11-13:
this is step 1 of at most two more data experiments for nano.

## Why
2026-11-13: nano + SARD (S) halved NOMAD false alarms at matched recall, but failed the guard because on
the held-out validation locations it fires on small white rocks on karst grass (1.46 vs 0.40 detections per
empty tile at 0.3). Question: does adding hard negatives of that kind fix this while keeping the NOMAD gain?

## Data
- SARD training tiles as on 2026-11-13 (`D:/datasets/sard_tiles/train.npz`, 4,505 tiles).
- Hard negatives: `mine_negatives.py` with `VOX_PHOTOS` = the 706 voxhod_g **training** photos only (never
  validation or test), scored by S3b (a SARD model, float, last epoch); per photo the **3** highest-scoring of
  40 empty candidate crops (no person box touches them), no random ones. About 2,100 tiles.
- Both appended to voxhod_g train via `VOX_EXTRA_TRAIN` (one combined npz). Val/test unchanged.

## Runs
- Arm R: nano recipe (width 0.5, lr 2e-3), **one run at a time**, float budget set to match the controls'
  optimizer steps (~27k): 15 min, then 10 min QAT. Seeds 1-5 (R1-R5).
- Control K: K1L, K2L, K3L, K4, K5 (unchanged). S1b-S5b reported as a descriptive third arm.

## Decision rule (same as 2026-11-13, fixed now)
Threshold per run: lowest threshold with val FA per tile <= 0.610. `arms_hier.py`, `ARMS_PAIRED=1`, 2000 draws.
- Primary A: HERIDAL test recall at 0.3 FA per tile, R - K: lower 95% > 0.
- Primary B: NOMAD FA per tile at the budget threshold, R - K: upper 95% < 0.
- Pass = A or B passes and no guard fails.
- Guards: HERIDAL recall at the budget threshold, lower 95% >= -2 points; WiSARD top-2 frame hit not lower by
  more than 3 points; HERIDAL FA at the budget threshold, point estimate not higher by more than 0.05.
- A pass means: retrain published nano and v3 on full HERIDAL + SARD + these negatives, under the same rule
  against the published files. Not "ship today".
