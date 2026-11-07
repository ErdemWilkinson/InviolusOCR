"""Structured channel pruning of a conv InviolusOCR model to a smaller width.

Builds the OCR_WIDTH model (OCR_ARCH=conv) that train.py would build, and fills
it with the most important channels of a trained wider model, so that
train.py can fine-tune it (OCR_WARM_START=<pruned .keras>). Because the target is exactly the
OCR_WIDTH architecture, a model trained from scratch at the same OCR_WIDTH is
the fair control: only the starting weights differ.

Channel importance (Network Slimming idea): |gamma| of the BatchNormalization
after each Conv2D. The 4 residual Conv1D blocks and the first Dense share one
channel stream (identity Add), so those channels are chosen once, from the sum
of the normalised |gamma| of the 4 Conv1D BatchNorms plus the normalised L1
norm of the Dense output columns.

  OCR_ARCH=conv OCR_WIDTH=0.7 python prune.py ../artifacts_D_1101/erdem_ocr.keras ../artifacts_P_1107/pruned_init.keras
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OCR_ARCH", "conv")
import tensorflow as tf  # noqa: E402

import train  # noqa: E402  (reads OCR_ARCH / OCR_WIDTH at import)


def by_type(model: tf.keras.Model, kind: type, exclude: type | None = None) -> list:
    return [layer for layer in model.layers
            if isinstance(layer, kind) and not (exclude and isinstance(layer, exclude))]


def top(scores: np.ndarray, n: int) -> np.ndarray:
    return np.sort(np.argsort(-scores)[:n])


def main() -> None:
    source_path, target_path = Path(sys.argv[1]), Path(sys.argv[2])
    source = tf.keras.models.load_model(source_path, compile=False)
    _, target = train.build_models()

    s_conv, t_conv = by_type(source, tf.keras.layers.Conv2D, tf.keras.layers.Conv1D), by_type(target, tf.keras.layers.Conv2D, tf.keras.layers.Conv1D)
    s_bn, t_bn = by_type(source, tf.keras.layers.BatchNormalization), by_type(target, tf.keras.layers.BatchNormalization)
    s_c1, t_c1 = by_type(source, tf.keras.layers.Conv1D), by_type(target, tf.keras.layers.Conv1D)
    s_dense, t_dense = by_type(source, tf.keras.layers.Dense), by_type(target, tf.keras.layers.Dense)
    assert len(s_conv) == len(t_conv) == 4 and len(s_c1) == len(t_c1) == 4 and len(s_bn) == len(t_bn) == 8
    assert len(s_dense) == len(t_dense) == 2

    # Conv2D body: each layer's output channels chosen on its own BN gamma.
    keep_in = np.arange(1)
    for i in range(4):
        n_out = t_conv[i].filters
        gamma = np.abs(s_bn[i].get_weights()[0])
        keep = top(gamma, n_out)
        w, b = s_conv[i].get_weights()
        t_conv[i].set_weights([w[:, :, keep_in, :][..., keep], b[keep]])
        t_bn[i].set_weights([v[keep] for v in s_bn[i].get_weights()])
        print(f"conv2d {i}: {w.shape[-1]} -> {n_out} channels, kept gamma share "
              f"{gamma[keep].sum() / gamma.sum():.1%}")
        keep_in = keep

    # Permute((2,1,3)) + Reshape((T, 2*C)): Dense input index = h * C + c, h in {0, 1}.
    c_src = s_conv[3].filters
    dense_in = np.concatenate([keep_in, keep_in + c_src])

    # Shared residual channel stream: Dense output + 4 Conv1D/BN outputs.
    n_seq = t_dense[0].units
    w_d, b_d = s_dense[0].get_weights()
    score = np.abs(w_d).sum(axis=0) / np.abs(w_d).sum(axis=0).mean()
    for bn in s_bn[4:]:
        g = np.abs(bn.get_weights()[0])
        score = score + g / g.mean()
    seq = top(score, n_seq)
    print(f"sequence stream: {w_d.shape[1]} -> {n_seq} channels, kept score share {score[seq].sum() / score.sum():.1%}")
    t_dense[0].set_weights([w_d[dense_in][:, seq], b_d[seq]])
    for s_layer, t_layer, s_norm, t_norm in zip(s_c1, t_c1, s_bn[4:], t_bn[4:]):
        w, b = s_layer.get_weights()
        t_layer.set_weights([w[:, seq, :][:, :, seq], b[seq]])
        t_norm.set_weights([v[seq] for v in s_norm.get_weights()])
    w_o, b_o = s_dense[1].get_weights()
    t_dense[1].set_weights([w_o[seq], b_o])

    print(f"parameters: {source.count_params():,} -> {target.count_params():,}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target.save(target_path)
    print(f"Wrote {target_path}")


if __name__ == "__main__":
    main()
