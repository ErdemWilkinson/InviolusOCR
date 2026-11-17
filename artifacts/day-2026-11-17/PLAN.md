# Day 2026-11-17 plan: two-stage nano → VoxScout (C), written before any C number was computed

Owner, after the nano freeze: "yollarını elinle sen teker teker oluşturamaz mısın ?" — build the proposed
paths one by one. Path 1 (this file): two stages. Path 2 (threshold calibration, needs a fresh step-matched
rerun) and path 3 (ASR teacher + distillation) come later, each with its own plan.

## Idea
Nano is a good ranker but raises many false alarms; VoxScout (EfficientNet-B0, separate project) has 6-9×
fewer false alarms but is 5.2 MB and slow. On the device: nano scans every tile, VoxScout only re-checks the
tiles nano flags. Detections come from VoxScout only.

## Construction (offline, from cached heatmaps, no training)
- Stage 1: nano runs K1L, K2L, K3L, K4, K5 (the control arm), heatmaps `D:/vox_g/{run}_{split}.npy`.
- Stage 2: VoxScout float heatmaps `D:/vox_g/T0_{split}.npy` (same tiles, 40x40).
- Gate per nano run, chosen on **val only**: tile passes if its nano maximum >= tau. tau = highest threshold
  (from `threshold_sweep.THRESHOLDS`) at which the passed val tiles still contain >= 97% of val people.
- Combined heatmap C_i = T0 heatmap on passed tiles, 0 elsewhere. C1..C5 pair with K1L..K5.

## Decision rule (same as T, 2026-11-16 night)
`arms_hier.py`, `ARMS_PAIRED=1`, reference GA1, val FA budget 0.610; arm C vs K.
- Primary A: HERIDAL recall at 0.3 FA/tile, lower 95% > 0. Primary B: NOMAD FA at budget threshold, upper 95% < 0.
- Guards: HERIDAL recall at budget lower 95% >= -2 pts; WiSARD top-2 frame hit not lower by > 3 pts;
  HERIDAL FA at budget not higher by > 0.05.
- Pass = A or B passes and no guard fails.

## Also reported (descriptive)
- Share of tiles sent to VoxScout on test, NOMAD, WiSARD → device cost estimate (nano 151 ms/tile on P4
  measured; VoxScout on P4 not measured).
- VoxScout alone on every tile (T0) vs C: what the gate costs.

## Caveats fixed now
- All five C runs share one VoxScout model, so its run-to-run spread is not in the interval.
- VoxScout numbers are float; int8 accuracy, P4 latency and P4 RAM are not measured. A pass here only
  justifies measuring those next, not shipping.
- If it ships, it ships as a VoxScout-project pipeline (backbone rule); the VoxHOD nano file is unchanged.

### Amendment ~18:15 (after the float C result, before any int8 number)
Owner: "devam et otopilot". The device would run VoxScout int8, not float. Step: run the published int8 file
(`voxhod_large_int8.tflite` = VoxScout int8, 5.2 MB) on val, test, WiSARD and NOMAD (`threshold_sweep.py`,
4 threads, name VS8), build C8_1..C8_5 with `two_stage.py` exactly as above (stage 2 = VS8), and score
C8 vs K with the same rule. Reported also: VS8 vs T0 (int8 loss). 40-tile check before this: correlation 0.86,
0.30 s per tile on the laptop CPU with 4 threads.
