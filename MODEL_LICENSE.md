# License for trained models

The [MIT License](LICENSE) in this repository covers the source code only.
It does not cover trained model files (`*.keras`, `*.tflite`) produced by
that code, whether or not they are distributed with this repository.

## Published models: CC BY-NC-SA 4.0

The models on [Hugging Face](https://huggingface.co/inviolus/InviolusOCR)
(D, 598 KB, and tiny, 171 KB) and every later model trained on the same data,
including the pruned 298 KB candidate, are released under
**[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)**:

- You may use, share and adapt them for non-commercial purposes.
- You must give credit to this project and to the datasets below.
- You must share adaptations under the same license.

The terms come from the training data:

| Dataset | License | What it adds to the model terms |
|---|---|---|
| Turkish Scene Text Recognition (TS-TR), Serdar Yıldız | CC BY-NC 4.0 | non-commercial |
| Synthetic Turkish Scene Text (STS-TR), Serdar Yıldız | CC BY-NC 4.0 | non-commercial |
| Turkish Word OCR, esengul3 | CC BY-SA 4.0 | share-alike |
| Turkish recognition 200k v4 (donut), turgutguvercin | CC BY-SA 4.0 | share-alike |
| Turkish OCR Text Image Dataset (Zenodo) | CC BY 4.0 | attribution |
| Synthetic lines rendered by `scripts/generate_synthetic.py` | this project | none |

## Not used: orkungedik

The OCR Turkish Word Dataset by orkungedik states no license. It was used
only in two unpublished October models (2026-10-10 and 2026-10-25). No
published model has been trained on it, and every model since 2026-10-31
excludes it (`OCR_EXCLUDE_SOURCES=orkungedik_lines`).

## A model without the non-commercial term

A model trained only on the synthetic lines, your own photos and the Zenodo
dataset is not bound by the non-commercial or share-alike terms. Train it by
leaving the TS-TR, STS-TR, esengul3 and donut manifests out of
`OCR_LABELS_CSV`.

## Attribution

- Turkish Scene Text Recognition (TS-TR), Serdar Yıldız, CC BY-NC 4.0.
  <https://www.kaggle.com/datasets/serdaryildiz/turkish-scene-text-recognition-dataset>
- Synthetic Turkish Scene Text Recognition (STS-TR), Serdar Yıldız, CC BY-NC 4.0.
  <https://www.kaggle.com/datasets/serdaryildiz/synthetic-turkish-scene-text-recognition-dataset>
- Turkish Word OCR, esengul3, CC BY-SA 4.0.
  <https://huggingface.co/datasets/esengul3/turkish-word-ocr>
- Turkish recognition 200k v4 (donut), turgutguvercin, CC BY-SA 4.0.
- Turkish OCR Text Image Dataset, CC BY 4.0.
  <https://zenodo.org/records/21923181>
