"""Unblock Quantization-Aware Training (QAT) for InviolusOCR conv models.

TF-MOT cannot fold BatchNormalization after Conv1D, which blocked QAT for
InviolusOCR Base (D), Small (P2) and Nano (Pt). Replacing the four residual
dilated Conv1D(k=5, d) blocks with mathematically identical Conv2D(k=(5,1), d=(d,1))
over (TIME_STEPS, 1, C) lets TF-MOT fold every Conv2D+BN cleanly while copying
100% of the trained float weights verbatim (zero accuracy loss before QAT).

Usage:
    # Smoke test weight transfer + QAT wrap + TFLite int8 conversion only:
    python scripts/qat_conv2d.py --smoke /root/ocr/artifacts_P2/erdem_ocr.keras /root/ocr/artifacts_P2/representative_images.npy /tmp/smoke_qat

    # Full QAT fine-tuning on GPU:
    OCR_LABELS_CSV=... OCR_FIXED_VALIDATION=... OCR_EXCLUDE_SOURCES=orkungedik_lines,real_lines \
      python scripts/qat_conv2d.py --src /root/ocr/artifacts_P2/erdem_ocr.keras --width 0.7 \
      --out /root/ocr/artifacts_P2_qat --minutes 12 --lr 2e-4 --batch 128
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import train as ocr_train


def build_conv2d_equivalent(width: float, static_batch: int | None = None) -> tf.keras.Model:
    """Mathematically identical to train.recognition_body(..., arch='conv'),
    with Conv1D(5, d) rewritten as Conv2D((5, 1), (d, 1)) and inference-only
    dropout omitted so TF-MOT quantizes every layer cleanly."""
    if static_batch is not None:
        inp = tf.keras.Input(batch_shape=(static_batch, ocr_train.IMAGE_HEIGHT, ocr_train.IMAGE_WIDTH, 1), name="image")
    else:
        inp = tf.keras.Input((ocr_train.IMAGE_HEIGHT, ocr_train.IMAGE_WIDTH, 1), name="image")

    w = lambda n: int(n * width)
    x = tf.keras.layers.Conv2D(w(32), 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(w(64), 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(w(96), 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)

    x = tf.keras.layers.Conv2D(w(128), 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)

    x = tf.keras.layers.Permute((2, 1, 3))(x)  # (T=40, H=2, C)
    x = tf.keras.layers.Reshape((ocr_train.TIME_STEPS, 2 * w(128)))(x)
    x = tf.keras.layers.Dense(w(128), activation="relu")(x)
    x = tf.keras.layers.Reshape((ocr_train.TIME_STEPS, 1, w(128)))(x)

    for dilation in (1, 2, 4, 1):
        y = tf.keras.layers.Conv2D(w(128), (5, 1), padding="same", dilation_rate=(dilation, 1))(x)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.ReLU()(y)
        x = tf.keras.layers.Add()([x, y])

    x = tf.keras.layers.Reshape((ocr_train.TIME_STEPS, w(128)))(x)
    logits = tf.keras.layers.Dense(ocr_train.NUM_CLASSES, activation="softmax", name="characters")(x)
    return tf.keras.Model(inp, logits)


def transfer_weights_conv1d_to_conv2d(src: tf.keras.Model, dst: tf.keras.Model) -> None:
    """Copy all trainable/BN weights in order, reshaping Conv1D kernels (k, Cin, Cout) -> (k, 1, Cin, Cout)."""
    src_w = src.get_weights()
    dst_w = dst.get_weights()
    assert len(src_w) == len(dst_w), f"Weight count mismatch: {len(src_w)} vs {len(dst_w)}"
    mapped = []
    for sw, dw in zip(src_w, dst_w):
        if sw.shape == dw.shape:
            mapped.append(sw)
        elif sw.ndim == 3 and dw.ndim == 4 and sw.shape[0] == dw.shape[0] and dw.shape[1] == 1 and sw.shape[1:] == dw.shape[2:]:
            mapped.append(sw[:, np.newaxis, :, :])
        else:
            raise ValueError(f"Incompatible weight shapes: src {sw.shape} vs dst {dw.shape}")
    dst.set_weights(mapped)


class StandaloneBNQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Fake-quantize output activations of BatchNormalization layers that follow ReLU
    (where TF-MOT cannot fold BN into the preceding Conv2D)."""

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False
        )]

    def get_config(self):
        return {}


def apply_qat(equiv: tf.keras.Model) -> tf.keras.Model:
    # Annotate standalone BN layers (the 4 body BNs after Conv2D(relu)); the 4 sequence
    # Conv2D(5,1)->BN->ReLU blocks are folded automatically by TF-MOT.
    body_bns = {l.name for l in equiv.layers if isinstance(l, tf.keras.layers.BatchNormalization)
                and l.inbound_nodes and isinstance(l.inbound_nodes[0].inbound_layers, tf.keras.layers.Conv2D)
                and l.inbound_nodes[0].inbound_layers.activation is not tf.keras.activations.linear}

    def annotate(layer):
        if layer.name in body_bns:
            return tfmot.quantization.keras.quantize_annotate_layer(layer, StandaloneBNQuantizeConfig())
        if isinstance(layer, tf.keras.layers.InputLayer):
            return layer
        return tfmot.quantization.keras.quantize_annotate_layer(layer)

    annotated = tf.keras.models.clone_model(equiv, clone_function=annotate)
    annotated.set_weights(equiv.get_weights())
    with tfmot.quantization.keras.quantize_scope({"StandaloneBNQuantizeConfig": StandaloneBNQuantizeConfig}):
        qat = tfmot.quantization.keras.quantize_apply(annotated)
    q_count = sum(1 for l in qat.layers if "quant" in l.name)
    print(f"TF-MOT QAT applied: {q_count}/{len(qat.layers)} layers quantized", flush=True)
    return qat


def export_qat_int8(qat_recog: tf.keras.Model, rep_images: np.ndarray, out_tflite: Path) -> int:
    def rep_gen():
        for img in rep_images:
            yield [img[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_recog)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    data = converter.convert()
    out_tflite.write_bytes(data)
    print(f"Wrote {out_tflite} ({len(data):,} bytes)", flush=True)
    return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rep", type=Path, default=None, help="representative_images.npy (for smoke or export)")
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--smoke", action="store_true", help="Only verify weight transfer + QAT wrap + int8 export")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    src = tf.keras.models.load_model(args.src, compile=False)
    equiv = build_conv2d_equivalent(args.width)
    transfer_weights_conv1d_to_conv2d(src, equiv)

    # Verify exact float equivalence
    rng = np.random.default_rng(0)
    dummy = rng.uniform(-1, 1, size=(4, ocr_train.IMAGE_HEIGHT, ocr_train.IMAGE_WIDTH, 1)).astype(np.float32)
    diff = float(np.max(np.abs(src(dummy, training=False).numpy() - equiv(dummy, training=False).numpy())))
    print(f"Float equivalence check max_abs_diff = {diff:.2e} (params={equiv.count_params():,})", flush=True)
    assert diff < 1e-2, f"Float mismatch too large: {diff}"

    # Apply TF-MOT QAT (folding sequence Conv2D+BN and fake-quantizing body ReLU->BN)
    qat_recog = apply_qat(equiv)
    print("TF-MOT QAT apply_qat() SUCCEEDED!", flush=True)

    if args.smoke:
        rep = np.load(args.rep) if args.rep else dummy
        export_qat_int8(qat_recog, rep[:50], args.out / "erdem_ocr_int8.tflite")
        print("SMOKE TEST PASSED!", flush=True)
        return

    # Load full OCR dataset using the exact same split logic as train.py
    rows = ocr_train.load_rows()
    images, labels, lengths, groups = ocr_train.make_arrays(rows)
    rng = np.random.default_rng(42)
    shuffle_idx = rng.permutation(len(rows))
    images, labels, lengths, groups = images[shuffle_idx], labels[shuffle_idx], lengths[shuffle_idx], groups[shuffle_idx]
    rows = [rows[i] for i in shuffle_idx]
    train_idx, val_idx = ocr_train.split_indices(rows, groups)

    input_lengths = np.full((len(rows), 1), ocr_train.TIME_STEPS, dtype=np.int32)
    label_lengths = lengths[:, np.newaxis]
    train_inputs = [images[train_idx], labels[train_idx], input_lengths[train_idx], label_lengths[train_idx]]
    val_inputs = [images[val_idx], labels[val_idx], input_lengths[val_idx], label_lengths[val_idx]]
    train_targets = np.zeros((len(train_idx), 1), dtype=np.float32)
    val_targets = np.zeros((len(val_idx), 1), dtype=np.float32)

    # Wrap qat_recog with CTC loss outside the quantized graph
    img_in = tf.keras.Input((ocr_train.IMAGE_HEIGHT, ocr_train.IMAGE_WIDTH, 1), name="image")
    lab_in = tf.keras.Input((ocr_train.MAX_TEXT_LENGTH,), dtype="int32", name="labels")
    in_len = tf.keras.Input((1,), dtype="int32", name="input_length")
    lab_len = tf.keras.Input((1,), dtype="int32", name="label_length")
    logits = qat_recog(img_in)
    ctc = tf.keras.layers.Lambda(
        lambda a: tf.keras.backend.ctc_batch_cost(a[1], a[0], a[2], a[3]), name="ctc_loss"
    )([logits, lab_in, in_len, lab_len])
    train_model = tf.keras.Model([img_in, lab_in, in_len, lab_len], ctc)
    train_model.compile(optimizer=tf.keras.optimizers.Adam(args.lr, clipnorm=5.0), loss=lambda _, v: v)

    fit_ds = ocr_train.augmented_dataset(train_inputs, train_targets, args.batch)
    callbacks = [
        ocr_train.TimeBudget(args.minutes),
        ocr_train.TimeCosine(args.lr, args.minutes),
    ]
    print(f"Starting QAT fine-tuning: width={args.width}, lr={args.lr}, batch={args.batch}, minutes={args.minutes}", flush=True)
    hist = train_model.fit(
        fit_ds,
        validation_data=(val_inputs, val_targets),
        epochs=50,
        callbacks=callbacks,
        verbose=2,
    )

    rep_idx = ocr_train.representative_indices(rows, train_idx)
    rep_imgs = images[rep_idx]
    np.save(args.out / "representative_images.npy", rep_imgs)
    (args.out / "alphabet.json").write_text(
        json.dumps({"alphabet": ocr_train.ALPHABET, "blank_id": ocr_train.BLANK_ID}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "history_qat.json").write_text(
        json.dumps({k: [float(v) for v in vs] for k, vs in hist.history.items()}, indent=2),
        encoding="utf-8",
    )
    # Export QAT full-int8 TFLite while qat_recog is in memory!
    export_qat_int8(qat_recog, rep_imgs, args.out / "erdem_ocr_int8.tflite")
    # Save original float equiv alongside the QAT int8 tflite so benchmark.py can compare float vs QAT int8
    equiv.save(args.out / "erdem_ocr.keras")
    print(f"Saved float equiv + QAT int8 tflite to {args.out}", flush=True)


if __name__ == "__main__":
    main()
