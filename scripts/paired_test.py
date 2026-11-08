"""Paired comparison of two int8 models on the same test rows (exact McNemar test).

Comparing two accuracies side by side ignores that both models read the same lines. The
paired test only looks at lines where exactly one model is right, which is far more
sensitive: a 1-point gap can be significant even though each accuracy has a ±1-2 point
margin on its own.

    python scripts/paired_test.py artifacts_A artifacts_B --sets tstr_test,heldout,donut_test --out result.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmark import load_image, predict_int8, test_sets
from evaluate import greedy_ctc_decode


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def predictions(folder: Path, images: np.ndarray) -> list[str]:
    alphabet = json.loads((folder / "alphabet.json").read_text(encoding="utf-8"))
    id_to_char = {index + 1: character for index, character in enumerate(alphabet["alphabet"])}
    return predict_int8(folder / "erdem_ocr_int8.tflite", images,
                        lambda logits: greedy_ctc_decode(logits, alphabet["blank_id"], id_to_char))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--sets", default="tstr_test,heldout,donut_test")
    parser.add_argument("--out")
    args = parser.parse_args()
    a, b = Path(args.a).resolve(), Path(args.b).resolve()
    result = {"a": a.name, "b": b.name}
    for name, rows in test_sets().items():
        if name not in args.sets.split(","):
            continue
        images = np.stack([load_image(Path(row["base"]) / row["image"]) for row in rows])
        right_a = np.array([p == row["text"] for p, row in zip(predictions(a, images), rows)])
        right_b = np.array([p == row["text"] for p, row in zip(predictions(b, images), rows)])
        only_a, only_b = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
        result[name] = {"rows": len(rows), "exact_a": float(right_a.mean()), "exact_b": float(right_b.mean()),
                        "only_a_right": only_a, "only_b_right": only_b, "mcnemar_p": mcnemar_p(only_a, only_b)}
        print(name, result[name], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
