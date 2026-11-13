# Night 2026-11-13 plan: does SARD as extra training data help nano? (written 03:25, before any data or result)

Owner approved on 2026-11-13 ("eğitimlere indirmelere vs başlayabilirsin"), GPU until 10:00.
Data: SARD, Zenodo record 24034563, CC BY 4.0, by the original authors (Sambolek, Ivasic-Kos,
Univ. Rijeka). 1,981 RGB drone frames, persons labeled. CC BY keeps VoxHOD's CC BY 4.0 possible.

## Question
Nano's weak point is telling people from look-alikes on terrain it has not seen (NOMAD, WiSARD).
Bigger nets and distillation did not help (2026-11-11/12). Does more varied **labeled** aerial data?

## Data preparation (`scripts/prepare_sard.py`)
- SARD's own split if it has one: only its train (+val) frames become training tiles; its test frames
  are kept out (descriptive check only). Without a split: split by frame order into 80/20 blocks.
- Scale matched to HERIDAL: per person tile, the crop size is chosen so the person's larger side lands
  on a value drawn from the HERIDAL training tiles' distribution (5/50/95%: 11/22/47 px).
- Per frame: 2 positive tiles and 1 random tile (same kinds as HERIDAL; no CopyPaste).
- Tiles appended to the voxhod_g training set via `VOX_EXTRA_TRAIN`; val/test unchanged (HERIDAL).

## Runs
- Arm S: nano recipe exactly as K (width 0.5, lr 2e-3, 30 min float then 10 min QAT from the last
  epoch, same code path) + SARD tiles. Seeds 1-5 (S1-S5), two runs in parallel on the GPU.
- Control: K1L, K2L, K3L, K4, K5 (seeds 1-5, same split, same budget, already trained 2026-11-11/12).
- Caveat: the controls ran on other nights. Same machine, code path and pairing on the GPU.

## Decision rule (fixed now)
Threshold per run: lowest threshold with val FA per tile <= 0.610 (GA1 at 0.20), as on 2026-11-12.
`arms_hier.py`, `ARMS_PAIRED=1` (K_i paired with S_i by seed), 2000 draws.
- **Primary A (finding people):** HERIDAL test recall at 0.3 false alarms per tile, S - K: lower 95% > 0.
- **Primary B (unseen terrain):** NOMAD false alarms per tile at the budget threshold, S - K: upper 95% < 0.
- **Pass = A or B passes and no guard fails.** (Two primaries: a pass on only one is reported as such.)
- Guards: HERIDAL recall at the budget threshold, lower 95% >= -2 points; WiSARD top-2 frame hit not
  lower by more than 3 points; HERIDAL FA at the budget threshold, point estimate not higher by > 0.05.
- A pass means "retrain the published nano and v3 on full HERIDAL + SARD", not "ship tonight".

## Also tonight (no decision)
- VoxScout int8 sweep (CPU) to replace "not measured" in its README with numbers.
- Descriptive: S and K on SARD's held-out test frames.

## Amendment 06:15 (no S result has been looked at; the S sweeps exist in the cache but are not read)
- S1-S4 ran two at a time and got only 21-25 epochs in 30 min (70-100 s/epoch, likely memory swap in
  WSL with the larger training set); S5 ran alone (114 epochs, 12 s/epoch). Controls K got ~110 epochs
  of 244 steps (~27k optimizer steps). S1-S5 are **not used** for the decision.
- Rerun as **S1b-S5b**: same seeds and recipe, **one at a time**, float budget **18 min** so that the
  number of optimizer steps matches the controls (~70 epochs x 385 steps ~ 27k), then 10 min QAT.
  The comparison is therefore at matched compute (steps), not matched wall-clock.
- Rule unchanged, K arm unchanged. If not all five finish by ~09:05, the decision uses the finished
  ones (paired by seed) and the report says so.
