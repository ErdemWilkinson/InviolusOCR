# Night 2026-11-16 plan: nano + empty drone tiles from Serbia (T), then ASR (written ~23:55, before any T data or result)

Owner: "tamam seçerek indir sonrasında bu gece eğitimlere başla olur mu eğer eğitim erken biterse asr ı buda +
yeni veri seti seç ardından bu yeni veri seti ile onu eğit", then "oto pilotta devam et".
This is the last data experiment of the nano stop rule (2026-11-13), replacing the OpenAerialMap step.

## Why
R (2026-11-16 day) failed like S: validation false alarms on white rocks in karst grass (1.32 per empty tile vs
0.40 for K). Hard negatives from HERIDAL's own training photos did not fix it, presumably because that terrain is
missing there. Question: do empty tiles from **other drone footage of similar terrain** fix it while keeping
the NOMAD gain?

## Data (licence verified on the Zenodo record: CC BY 4.0)
- "Comprehensive dataset from high resolution UAV land cover mapping of diverse natural environments in Serbia",
  Zenodo 14169485, CC BY 4.0. DJI Inspire 1, 4000x3000 nadir RGB, GSD 3-4 cm, 27 sites. Only chosen RGB photos are
  fetched from the 22.8 GB RAR with HTTP range requests (scratchpad `serbia/rar_get.py`).
- Site choice by eye from 1-3 preview photos per site, **before** any model scores them: keep mountain/upland
  pasture, rocky grassland, scrub, meadow; drop sites dominated by water, crops, settlements or roads. About
  100-200 photos.
- Tiles: crop sizes 160/240/320 px (half of HERIDAL's 320/480/640, since GSD is about twice HERIDAL's), resized to
  160. Per photo 40 random candidates, scored by S3b (as for R); keep the 3 highest-scoring + 1 random. The photos
  are not labeled, so the highest-scoring crops are the ones most likely to hold a real person: **every kept
  hard negative is inspected on contact sheets and any crop with a person, or anything I cannot rule out, is
  dropped** before training. Number dropped is reported.
- Training set of arm T = R's set (voxhod_g train + SARD + karst negatives) + the Serbia tiles. Val/test unchanged.

## Runs
- Arm T: nano recipe, one run at a time, 15 min float + 10 min QAT, seeds 1-5 (T1-T5). **No CPU pinning** (owner
  asleep), so steps should match K (~27k); steps are reported.
- Control K: K1L, K2L, K3L, K4, K5. R1-R5 and S1b-S5b reported descriptively.

### Amendment 01:05 (before any T result; T1 was at epoch ~20 of its first run)
T1 ran at 25-55 s per epoch instead of R2's 14 s: two other processes share the GPU (another session's
python and a qemu emulator, not ours, left running) and the GPU sits at about half clock. At 15 min T would get
about 13k steps, half of K. Change: float budget **30 min** for every T run (T1 restarted), QAT unchanged at
10 min. Steps are reported per run; if T exceeds K's ~27k by more than 25% that is stated as a bias in T's favour.
The serbia_neg inspection dropped 92 of 764 tiles (people, possible people, vehicles with people, unclear
objects); 672 kept.

### Amendment 01:48 (before looking at any T sweep)
Correction to the 01:05 note: after the restart T1 ran at 16-20 s per epoch, so the slow start was mostly my own
CPU load at the time (photo downloads, tile mining, an ASR smoke test), not the other GPU processes. T1 finished
90 epochs = about 42k steps, **1.55x K's ~27k**. All T runs keep the 30 min budget so the arm is uniform.
Interpretation fixed now: if T fails the gate, the conclusion stands (it failed with more training than K);
if T passes, it is **not** accepted as a pass until a step-matched rerun (15 min at normal speed) confirms it.

## Decision rule (unchanged, same as 2026-11-13 and 2026-11-16)
Threshold per run: lowest threshold with val FA per tile <= 0.610. `arms_hier.py`, `ARMS_PAIRED=1`, 2000 draws.
- Primary A: HERIDAL test recall at 0.3 FA per tile, T - K: lower 95% > 0.
- Primary B: NOMAD FA per tile at the budget threshold, T - K: upper 95% < 0.
- Pass = A or B passes and no guard fails.
- Guards: HERIDAL recall at the budget threshold, lower 95% >= -2 points; WiSARD top-2 frame hit not lower by
  more than 3 points; HERIDAL FA at the budget threshold, point estimate not higher by more than 0.05.
- Also reported: detections per empty validation tile at 0.3 (white rocks), T vs K/R/S.
- Fail = nano is frozen (stop rule). Pass = retrain the published nano and v3 on full data under the same rule.

## ASR, after T if time is left (owner asked: prune + new dataset + train)
The ASR README (2026-11-07) says v11 underfits, so pruning is expected to hurt. Done anyway as a measured
experiment, since the owner asked: prune v11, fine-tune on v11's data plus a new permissively licensed Turkish
dataset (CC0 / CC BY only; no orkungedik, no NC/SA, no unlicensed), then compare with v11 on the same 500 Common
Voice and 500 ISSAI test clips (`scripts/evaluate_asr.py`, greedy). Nothing is published tonight; results go to
`RAPOR.md`. Stop at 10:00.
