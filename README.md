# InviolusOCR: tiny Turkish OCR for microcontrollers

**Türkçe:** Mikrodenetleyiciler için küçük Türkçe OCR modeli. Tek satır basılı
Türkçe metni okur. Yayındaki int8 TensorFlow Lite dosyası 598 KB, küçük sürümü
171 KB. 298 KB'lık budanmış bir ara sürüm deneme aşamasında. Grafikler, sürüm
geçmişi ve yol haritası aşağıda.

*Keywords: Turkish OCR, Türkçe OCR, tiny OCR, text recognition, CTC, TFLite,
int8, TinyML, ESP32, microcontroller.*

**InviolusOCR** is a compact, full-int8 CNN-CTC recognizer for a single
horizontal line of printed Turkish text, sized to run on a
microcontroller-class embedded target. It started as part of the
[makeshift-flipper](https://github.com/ErdemWilkinson/makeshift-flipper)
project and was split out because it is a separate concern (a Python and
TensorFlow training pipeline, not ESP-IDF firmware) with its own, much larger
dataset.

**Status:** the training pipeline works end to end, and speed and memory are
measured on an ESP32-P4 board. It is not integrated into any firmware yet.

| Model | Params | int8 file | Held-out | TS-TR | Donut | Where |
|---|---:|---:|---:|---:|---:|---|
| **D** | 560,954 | 598 KB | **77.3%** | **56.2%** | **46.5%** | [Hugging Face](https://huggingface.co/inviolus/InviolusOCR) (main) |
| P2, pruned from D | 274,450 | 298 KB | 72.3% | 46.0% | 33.2% | candidate, not published |
| **tiny** | 144,330 | 171 KB | 64.4% | 32.6% | 20.1% | [Hugging Face](https://huggingface.co/inviolus/InviolusOCR) |

Scores are the share of lines read exactly right by the int8 file. See
[Where the model stands](#where-the-model-stands) for the test sets,
[How it got here](#how-it-got-here) for every version, and
[Roadmap](#roadmap) for what comes next.

## Where the model stands

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ocr-models-dark.svg">
  <img alt="Exact-line accuracy of the int8 files of D, P2 and tiny on three test sets" src="docs/ocr-models-light.svg">
</picture>

All models are scored by `scripts/benchmark.py` on the same rows:

- **Held-out lines:** 12,093 lines from the training sources (synthetic,
  Zenodo, TS-TR, esengul3, STS-TR) that no model trained on. 95% noise margin
  about ±0.9 points.
- **TS-TR scene text:** the official TS-TR test split, 2,078 real photos of
  signs and shop fronts, never trained on. Noise margin about ±2.0 points.
- **Donut:** 2,563 printed-document rows (dates, amounts, headings) from the
  donut 200k test list.

Character error rate of the int8 files:

| Model | Held-out | TS-TR | Donut |
|---|---:|---:|---:|
| D | 10.0% | 18.8% | 12.6% |
| P2 | 11.8% | 23.9% | 18.1% |
| tiny | 14.1% | 31.3% | 24.1% |

What this means:

- **Clean printed lines work.** D reads about three in four held-out lines
  exactly; most errors are one wrong character.
- **Real scene text is the weak spot.** Even D reads only 56% of TS-TR photos
  exactly.
- **Smaller costs accuracy, but not evenly.** P2 keeps 94% of D's held-out
  score at half the size. tiny keeps 83% at 29% of the size.
- **None of them meets the acceptance targets yet** (CER ≤ 5%, exact ≥ 80%;
  see [Acceptance targets](#acceptance-targets)).

## How it got here

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ocr-lineage-dark.svg">
  <img alt="Family tree of the model versions from 2026-10-09 to 2026-11-07 with their held-out int8 accuracy" src="docs/ocr-lineage-light.svg">
</picture>

The biggest step was a change of architecture, not more training. The October
models used a BiLSTM and read up to 80% of held-out lines on a computer, but
lost 26 to 48 points when converted to the int8 file a device runs. Replacing
the BiLSTM with residual dilated 1-D convolutions (model B) cut that loss to
about 1 point:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ocr-int8-gap-dark.svg">
  <img alt="Held-out accuracy of the float model and of its int8 file for every version" src="docs/ocr-int8-gap-light.svg">
</picture>

The goal is the smallest Turkish OCR that still works, so what counts is the
best accuracy at each file size. Every current model is both smaller and more
accurate than every October model:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ocr-size-frontier-dark.svg">
  <img alt="int8 file size against held-out accuracy for every measured version" src="docs/ocr-size-frontier-light.svg">
</picture>

### Version history

All rows use the same test rows. Earlier write-ups of each run, with their own
and no longer comparable test sets, were removed from this README to avoid
confusion; they are in the git history up to commit `d210a8b`.

| Version | Date | What changed | Params | int8 | Held-out | TS-TR | Donut | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| LSTM | 2026-10-09 | TS-TR scene text added | 631,226 | 665 KB | 32.9% | 33.5% | n/m | was on Hugging Face as `erdem-ocr-tr-tinyml`, replaced |
| LSTM | 2026-10-10 | augmentation, 82,814 word crops | 631,226 | 665 KB | 39.5% | 42.7% | n/m | never published (orkungedik data, no licence) |
| LSTM | 2026-10-25 | 35 more epochs | 631,226 | 665 KB | 32.1% | 48.7% | 6.7% | never published (same reason) |
| A, LSTM | 2026-10-31 | from 10-09 weights, orkungedik removed | 631,226 | 665 KB | 48.2% | 41.6% | n/m | replaced by B |
| B, conv | 2026-10-31 | conv sequence head instead of LSTM | 560,954 | 598 KB | 78.4% | 58.2% | 9.3% | replaced by D |
| **D, conv** | 2026-11-01 | B + 47,613 donut document lines | 560,954 | 598 KB | 77.3% | 56.2% | 46.5% | **on Hugging Face (main)** |
| E, F | 2026-11-02 | D trained 5 more hours | 560,954 | 598 KB | 77.1% | 55.5–56.0% | 45.9–46.5% | dead end |
| **tiny** | 2026-11-02 | half width, from scratch | 144,330 | 171 KB | 64.4% | 32.6% | 20.1% | **on Hugging Face** |
| K, K2 | 2026-11-04/06 | tiny distilled from D as teacher | 144,330 | 171 KB | 62.4–63.6% | 31.4–32.7% | n/m | dead end |
| P | 2026-11-07 | D pruned to 70% width, 8 epochs | 274,450 | 298 KB | 71.8% | 44.5% | 31.7% | replaced by P2 |
| S | 2026-11-07 | P's architecture from scratch (control) | 274,450 | 298 KB | 65.3% | 34.4% | 20.6% | control only |
| **P2** | 2026-11-07 | P + 5 epochs on a GPU | 274,450 | 298 KB | 72.3% | 46.0% | 33.2% | **candidate** |

n/m: not measured.

What worked and what did not:

- **Worked:** real scene text (TS-TR) in training; the conv sequence head;
  the donut document lines (printed documents from 9% to 46%); pruning.
- **Did not work:** more epochs of the same training (10-25 over 10-10, E and
  F over D); distilling D into tiny (K, K2 scored below tiny trained alone).

### Latest experiment: pruning D to half its size

Does cutting channels out of a trained model give a better small model than
training that small model from scratch? Liu et al. (2019, "Rethinking the
Value of Network Pruning") found that often it does not, so this run has a
control.

- **P:** `scripts/prune.py` builds the `OCR_WIDTH=0.7` model and fills it
  with the channels of D that have the largest BatchNorm |gamma| (the Network
  Slimming rule). The four residual Conv1D blocks share one channel stream, so
  their channels are chosen together. Right after the cut the model reads
  0.1% of TS-TR lines, so the fine-tuning does the work, not the cut alone.
- **S:** the same architecture from random weights.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ocr-pruning-channels-dark.svg">
  <img alt="Importance scores of D's 128 sequence channels, sorted; the top 89 are kept and 39 are cut" src="docs/ocr-pruning-channels-light.svg">
</picture>

The scores are close together (6.5 at the top, 4.1 at the bottom): D had no idle channels,
every one did some work. That is why the model fails right after the cut and needs fine-tuning.

- Both: the same data and validation rows as D, learning rate 1e-3 with
  cosine decay, augmentation, 7 hours each on a CPU. That was only 8 epochs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/ocr-prune-vs-scratch-dark.svg">
  <img alt="int8 accuracy of the pruned model P and the from-scratch model S on three test sets" src="docs/ocr-prune-vs-scratch-light.svg">
</picture>

- **Pruning won clearly at this budget:** 6.5 points on held-out lines, 10.1
  on TS-TR and 11.1 on donut, well above the noise margins.
- **But the runs were short.** S was still improving faster at the end, so a
  longer run may close part of the gap. That is the next experiment.
- **P2** is P trained 5 more epochs on a GPU (learning rate 5e-4, cosine):
  +0.5, +1.5 and +1.5 points. Small gains, within the noise on held-out and
  TS-TR, but in the same direction on all three sets.
- **int8 loss of P2:** 1.0 point on held-out, 2.7 on TS-TR, 4.1 on donut.

Results: `artifacts/prune-2026-11-07/benchmark_*.json`.

## Roadmap

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/roadmap-ocr-dark.svg">
  <img alt="InviolusOCR roadmap: next steps, this week and later, each with its gate" src="docs/roadmap-ocr-light.svg">
</picture>

Rule for every step: **the device model does not grow.** 598 KB is the
largest file; larger models are only allowed as teachers during training.
A step ships only if it passes its gate.

| Step | Why | Gate to ship | Status |
|---|---|---|---|
| Train P and S longer on the GPU (about 1 hour each) | Find out whether pruning still wins with a long budget | P ahead of S by ≥ 2 points on TS-TR | next |
| Prune D to tiny's size (width 0.5) | The current tiny was trained from scratch; pruning beat that at 298 KB | At 171 KB: held-out > 64.4% and TS-TR > 32.6% | planned |
| Quantization-aware training for the pruned model | P2 loses 4.1 points on donut when converted to int8 | float-to-int8 loss ≤ 2 points on every set | planned |
| Publish a 298 KB model on Hugging Face | Fill the gap between tiny and D | The gates above, plus a measurement on the ESP32-P4 | after the above |
| Feed the GPU faster | The GPU is only about 30% busy; preparing batches is the bottleneck | Epoch time below 100 s (now 155 s) | open |
| More real scene-text data | TS-TR is the weakest set | The licence allows it; no orkungedik data | open |

Not planned again: longer training of the same model, and distilling D into
tiny (see the version history).

Training moved from the CPU to a laptop GPU on 2026-11-07 (WSL2, Ubuntu
22.04, TensorFlow 2.15.1, RTX 4050 6 GB). One epoch of the 298 KB model takes
155 s on the GPU against 46 to 59 minutes on the CPU, so each step above is a
matter of hours.

## Why a custom model instead of an existing OCR package

This was a deliberate choice, made before any training code was written;
see [DECISION_NOTES.md](DECISION_NOTES.md) for the full reasoning. Short
version: PaddleOCR/PP-OCR and similar general OCR stacks assume a phone or
a real application processor, not a memory-constrained MCU; Tesseract isn't
a TinyML model at all; and per-character classification breaks on kerning,
accents, and variable spacing. A small grayscale CNN + CTC sequence
recognizer for one pre-cropped text line is the smallest model that still
handles variable-length Turkish text correctly.

## Model

- **Input:** `160 x 32 x 1` grayscale image (one cropped, mostly horizontal
  text line), scaled to `[-1, 1]`.
- **Architecture** (`OCR_ARCH=conv` in `scripts/train.py`): 4 conv +
  batch-norm + pool blocks, a dense projection, four residual dilated 1-D
  convolutions (dilation 1, 2, 4, 1), and a dense softmax over the character
  set. `OCR_WIDTH` scales every layer (D: 1.0, P2: 0.7, tiny: 0.5). The older
  BiLSTM head is still available as `OCR_ARCH=lstm`, which is the default.
- **Output:** per-timestep character logits (40 timesteps, `IMAGE_WIDTH // 4`),
  including a CTC blank class.
- **Character set** (`ALPHABET` in `scripts/train.py`): digits, ASCII
  upper/lowercase, the Turkish letters `ç ğ ı İ ö ş ü Ç Ğ Ö Ş Ü`, a space,
  and `.,:;!?-_/()'%`. Max decoded text length: 24 characters.
- **Decoder:** greedy CTC (argmax per timestep, collapse repeats and blanks),
  deliberately no beam search, to keep post-processing trivial on-device.
- **Quantization:** full int8 (weights, activations, input, output) via
  `scripts/export_tflite.py`.
- **Not included:** text line detection and cropping. Classical
  preprocessing (thresholding, morphology, connected components) is expected
  to produce the line crop before this model runs; see `DECISION_NOTES.md` §4.

## Repository layout

```text
ocr/
├── scripts/
│   ├── generate_synthetic.py      # builds data/lines/synthetic/ + labels.csv rows
│   ├── prepare_real_lines.py      # crops real document photos into labeled lines
│   ├── prepare_code_real_lines.py # line crops for code-style text
│   ├── import_zenodo_dataset.py   # Zenodo printed lines -> data/zenodo_lines/
│   ├── import_tstr_dataset.py     # TS-TR scene-text crops -> data/tstr_lines/
│   ├── prepare_donut.py           # donut 200k document lines -> donut manifest
│   ├── prepare_plates_test.py     # licence-plate test crops (evaluation only)
│   ├── train.py                   # trains the model, writes artifacts/
│   ├── prune.py                   # structured channel pruning of a trained model
│   ├── evaluate.py                # CER + exact-line accuracy on held-out groups
│   ├── benchmark.py               # scores models, float and int8, on the shared test rows
│   ├── make_representative.py     # rebuilds the int8 calibration sample set
│   ├── export_tflite.py           # float .keras -> full-int8 .tflite
│   ├── evaluate_tflite.py         # scores the exported .tflite on the held-out split
│   ├── make_readme_charts.py      # draws the charts in docs/ (uses svgkit.py)
│   └── gradio_demo.py             # local demo UI for trying the model on an image
├── benchmarks/esp32-p4/           # ESP-IDF firmware used for the on-device measurement
├── data/
│   ├── wordlists/tr_50k.txt       # tracked: small (~700KB) input corpus
│   └── ...                        # NOT tracked: generated/imported images and manifests
├── artifacts/
│   ├── alphabet.json              # tracked: the character set, hand-decided
│   ├── night-*/, prune-*/         # tracked: benchmark JSON of each run
│   └── *.keras, *.tflite, *.npy   # NOT tracked: trained model outputs
├── docs/                          # tracked: the charts in this README
├── DECISION_NOTES.md              # the model-selection writeup
├── MODEL_LICENSE.md               # terms for trained models (separate from the MIT code license)
└── requirements-train.txt
```

Only code, the wordlist corpus, the alphabet, benchmark results and charts
are tracked. All data and trained models are gitignored; trained models are
published on Hugging Face instead.

## Dataset layout

```text
ocr/data/lines/<split>/<id>.png
ocr/data/labels.csv
```

`labels.csv` columns: `image,text,group`.

```csv
image,text,group
lines/train/000001.png,İstanbul,generated_font_01
lines/train/000002.png,şifre 123,phone_photo_01
```

`group` drives the train/validation split (`GroupShuffleSplit` in
`train.py`, 15% held out): images from the same font/background/source
group must never appear in both splits, or validation numbers become
optimistic. `train.py` rejects any row whose text is empty, over 24
characters, or contains a character outside `ALPHABET`.

### Building the dataset

Each source has its own manifest; `OCR_LABELS_CSV` combines them.

1. **Synthetic** (`scripts/generate_synthetic.py`): renders 18,000 lines
   from `data/wordlists/tr_50k.txt` with randomized fonts, backgrounds, blur
   and perspective (fixed seed). Writes `data/labels.csv`.
2. **Real photos** (`scripts/prepare_real_lines.py`): crops photographed
   pages (kept locally, never committed) into `data/real_lines/` and
   `data/real_labels.csv`. Not used by the current models: the crops were
   regenerated on 2026-10-09 and no longer match their labels.
3. **Zenodo** (`scripts/import_zenodo_dataset.py`): the printed subsets of
   the Turkish OCR Text Image Dataset.
4. **TS-TR** (`scripts/import_tstr_dataset.py`): real scene text. The
   official train split becomes `data/tstr_labels.csv` (5,206 crops); the
   official test split is kept apart as `data/tstr_test_labels.csv` (2,078).
5. **Donut** (`scripts/prepare_donut.py`): rows of the recommended train list
   of Turkish recognition 200k v4 with 1 to 24 characters from the alphabet;
   its test list gives the 2,563-row `donut_test` set.
6. **esengul3** (45,000 crops) and **STS-TR** (7,814 crops) were imported
   into the same `image,text,group` manifest format.

### Third-party data and licenses

None of this data is redistributed in this repository; each source keeps
its own license, and attribution is required where the license says so.

| Dataset | Used for | License | Source |
|---|---|---|---|
| Turkish word-frequency list (`data/wordlists/tr_50k.txt`) | Text for synthetic lines | Original license of the list | Tracked in this repo |
| Turkish OCR Text Image Dataset | Training (printed lines) | CC BY 4.0 | [Zenodo record 21923181](https://zenodo.org/records/21923181) |
| Turkish Scene Text Recognition (TS-TR), Serdar Yıldız | Training and evaluation (real scene text) | **CC BY-NC 4.0** | [Kaggle](https://www.kaggle.com/datasets/serdaryildiz/turkish-scene-text-recognition-dataset), [MViT-TR repository](https://github.com/serdaryildiz/MViT-TR) |
| Synthetic Turkish Scene Text (STS-TR), Serdar Yıldız | Training (synthetic scene text) | **CC BY-NC 4.0** | [Kaggle](https://www.kaggle.com/datasets/serdaryildiz/synthetic-turkish-scene-text-recognition-dataset) |
| Turkish Word OCR, esengul3 | Training (printed/rendered words) | CC BY-SA 4.0 | [Hugging Face](https://huggingface.co/datasets/esengul3/turkish-word-ocr) |
| Turkish recognition 200k v4 (donut), turgutguvercin | Training and evaluation (printed documents) | CC BY-SA 4.0 | Hugging Face |
| OCR Turkish Word Dataset, orkungedik | **Not used since 2026-10-31** (only in the unpublished 10-10 and 10-25 models) | No license listed | [Hugging Face](https://huggingface.co/datasets/orkungedik/ocr_turkish_word_dataset) |

TS-TR and STS-TR are non-commercial and esengul3 and donut are share-alike,
so the published models are CC BY-NC-SA 4.0; see
[MODEL_LICENSE.md](MODEL_LICENSE.md). The orkungedik dataset states no
license, so no model trained on it is published.

## Training pipeline

From this directory, in a Python 3.11 environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt

python scripts\generate_synthetic.py     # builds the synthetic split, writes data\labels.csv

# The current conv model on all sources (PowerShell uses ; as the separator, Linux uses :)
$env:OCR_ARCH = "conv"
$env:OCR_LABELS_CSV = "data\labels.csv;data\zenodo_labels.csv;data\tstr_labels.csv;<donut>;<esengul3>;<stsr>"
$env:OCR_EXCLUDE_SOURCES = "orkungedik_lines,real_lines"
$env:OCR_AUGMENT = "1"; $env:OCR_COSINE = "1"
python scripts\train.py

python scripts\export_tflite.py          # writes artifacts/erdem_ocr_int8.tflite
python scripts\benchmark.py --model D=artifacts --sets tstr_test,heldout,donut_test
```

Environment variables read by `train.py`:

| Variable | What it does |
|---|---|
| `OCR_ARCH` | `conv` (current models) or `lstm` (default, the October models) |
| `OCR_WIDTH` | Layer-width multiplier: 1.0 for D, 0.7 for P2, 0.5 for tiny |
| `OCR_LABELS_CSV` | List of manifests, separated by `;` on Windows and `:` on Linux |
| `OCR_EXCLUDE_SOURCES` | Drops whole sources by their first folder name, e.g. `orkungedik_lines` |
| `OCR_ARTIFACTS` | Output folder (default `artifacts/`) |
| `OCR_WARM_START` | Continue from an existing model's weights |
| `OCR_LR`, `OCR_COSINE` | Initial learning rate (default 1e-3) and cosine decay over the time budget |
| `OCR_AUGMENT` | Random shift, contrast, brightness and noise on training crops |
| `OCR_MAX_MINUTES` | Time budget; keeps the weights of the best validation epoch |
| `OCR_FIXED_VALIDATION` | Keeps an earlier run's held-out rows held out (`split_manifest.json`) |
| `OCR_PATIENCE` | Early-stopping patience in epochs (default 10) |
| `OCR_TEACHER`, `OCR_KD_WEIGHT` | Distillation from a frozen teacher model (tried, did not help) |
| `OCR_THREADS` | CPU threads for TensorFlow |

`export_tflite.py` rebuilds the model from the same `OCR_ARCH`, `OCR_WIDTH`
and `OCR_ARTIFACTS`, so set them as for training.

Pruning a trained model to a smaller width, then fine-tuning it:

```bash
OCR_ARCH=conv OCR_WIDTH=0.7 python scripts/prune.py artifacts_D/erdem_ocr.keras artifacts_P/pruned_init.keras
OCR_ARCH=conv OCR_WIDTH=0.7 OCR_WARM_START=artifacts_P/pruned_init.keras OCR_ARTIFACTS=artifacts_P python scripts/train.py
```

**GPU training:** TensorFlow 2.11 and later has no GPU support on native
Windows, so the GPU runs use WSL2 with Ubuntu 22.04 and
`pip install "tensorflow[and-cuda]==2.15.1"`. Two pitfalls: a `.keras` file
saved on Windows cannot be loaded on Linux with Keras 2.15 (its weight paths
use `\`), so move models between the two as `.h5`; and TensorFlow on Windows
cannot open an `.h5` file under a folder with non-ASCII characters.

## Acceptance targets

Before any of this is treated as ready to integrate into firmware, on a
source/font/background-independent held-out set:

- Turkish character error rate (CER): ≤ 5%
- Exact-line accuracy: ≥ 80% for short controlled lines
- No unsupported operators in the target embedded TFLite runtime
- Model + tensor arena size measured on real target hardware, not guessed
- First-pass latency target: ≤ 1 second per cropped line

Current state: no model meets the CER and exact-line targets yet (D: CER
10.0%, 77.3% exact on held-out lines). Operators, arena size and latency are
measured on an ESP32-P4 below: tiny meets the latency target, D does not.

## On-device measurement (ESP32-P4)

Measured on 2026-11-06 on one ESP32-P4 board (rev v1.3, 360 MHz, 32 MB flash, 32 MB PSRAM) with ESP-IDF 5.3.5 and `esp-tflite-micro` 1.4.2. Each row is the mean of 20 runs after 2 warm-up runs; the published files were measured byte for byte. One inference is one 160x32 grayscale line. The makeshift-flipper itself is an ESP32-C6; the P4 was a separate development board used only for this measurement.

| Model | File | Arena used | Latency, arena in internal SRAM | Latency, arena in PSRAM (200 MHz) |
|---|---:|---:|---:|---:|
| InviolusOCR tiny | 171 KB | 172 KB | 413 ms | 417 ms |
| InviolusOCR D | 598 KB | 339 KB | 1365 ms | 1387 ms |

The weights stay in flash; the arena only holds the activations. Both models fit in internal SRAM. P2 (298 KB) has not been measured on the device yet.

What this does and does not show:

- Random int8 input: this is **speed and memory only**. The output was not compared with the PC
  interpreter, so correctness on the device is unchecked.
- Only the model inference is timed, not image decoding, resizing, CTC decoding.
- It is an ESP32-P4, not an ESP32-S3. The numbers do not carry over to the S3, which has other cores.
- Whether ESP-NN's accelerated kernels are active for the P4 was not checked, so the speed may improve.
- One board. Source and raw output: [benchmarks/esp32-p4](benchmarks/esp32-p4).

## Hardware integration status

This pipeline was deliberately built **before** committing to firmware
integration, because on-device OCR has a hardware dependency this repo
alone can't resolve. The makeshift-flipper is now a single ESP32-C6 (the
earlier ESP32-P4 board is no longer part of the project), so there is
currently **no chosen target board for this model**, and **the current
makeshift-flipper firmware has no camera driver, camera pin assignment,
frame buffer, or TinyML runtime integration at all.** The device's only
image-adjacent hardware is a 240x240 ST7789 SPI LCD, which is an output-only
display and cannot supply an image to this model.

Before writing any firmware integration code, these need to be decided and
confirmed against a real board:

- Which board runs the model. The ESP32-C6 has no PSRAM and no camera
  interface in the current build, and whether this model fits in its RAM
  next to the existing firmware has not been measured
- Camera module + sensor interface (DVP/MIPI) and actual free pin budget
- Whether the camera can deliver grayscale or only RGB565 frames
- Flash/RAM headroom left after the existing firmware links
- Target embedded runtime for this ESP-IDF version: LiteRT/TFLite Micro vs.
  ESP-DL
- The real use case this needs to serve: printed labels/cards vs. general
  scene text (this model is scoped for the former; see
  `DECISION_NOTES.md` §5 for what it explicitly does *not* promise)

## Related repositories

- [makeshift-flipper](https://github.com/ErdemWilkinson/makeshift-flipper):
  the ESP32-C6 handheld firmware project this model was originally scoped for
- [VoxHOD](https://github.com/ErdemWilkinson/VoxHOD): the sibling aerial
  person detector, built with the same toolchain
- [turkish-asr-whisper](https://github.com/ErdemWilkinson/turkish-asr-whisper):
  the sibling TinyML pipeline for offline Turkish speech commands

## License

Code in this repository is released under the [MIT License](LICENSE). The word-frequency list in `data/wordlists/` is third-party data and keeps its original license.

Trained model files are not covered by the MIT License. Models trained with
non-commercial data are for non-commercial use only; see
[MODEL_LICENSE.md](MODEL_LICENSE.md) and
[Third-party data and licenses](#third-party-data-and-licenses).
