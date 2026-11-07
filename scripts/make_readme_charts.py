"""Draw the InviolusOCR README charts into docs/ (light and dark SVG).

All numbers are exact-line accuracy measured by scripts/benchmark.py on the same rows:
12,093 held-out lines, the 2,078-crop TS-TR test split and 2,563 donut test rows.
Sources: artifacts/night-2026-11-01/benchmark.json (10-09, 10-25, B, D),
the 2026-10-31 README table (10-10, A), night-2026-11-02/benchmark_T.json (tiny),
night-2026-11-04/benchmark_K.json, night-2026-11-06/benchmark_K2.json,
prune-2026-11-07/benchmark_{P,S,P_donut,S_donut,P2}.json.

    python scripts/make_readme_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svgkit import channel_bars, esc, fmt, dumbbell, grouped_bars, header, roadmap, write  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"

# name, int8 KB, arch, held-out float, held-out int8
VERSIONS = [
    ("2026-10-09", 665, "LSTM", 58.8, 32.9),
    ("2026-10-10", 665, "LSTM", 77.2, 39.5),
    ("2026-10-25", 665, "LSTM", 80.0, 32.1),
    ("A, 10-31", 665, "LSTM", 76.9, 48.2),
    ("B, 10-31", 598, "conv", 79.4, 78.4),
    ("D, 11-01", 598, "conv", 78.4, 77.3),
    ("tiny, 11-02", 171, "conv", 66.3, 64.4),
    ("P, 11-07", 298, "conv", 72.8, 71.8),
    ("P2, 11-07", 298, "conv", 73.3, 72.3),
]

# Importance of D's 128 shared sequence channels (normalised Dense L1 norm + sum of normalised
# BatchNorm |gamma| of the 4 Conv1D blocks), sorted; scripts/prune.py keeps the top 89 (width 0.7).
SEQ_SCORES = [6.53, 6.36, 6.06, 5.99, 5.95, 5.85, 5.84, 5.73, 5.69, 5.67, 5.62, 5.61, 5.56, 5.55, 5.52, 5.51, 5.47, 5.43, 5.42, 5.4, 5.39, 5.38, 5.35, 5.34, 5.33, 5.33, 5.32, 5.32, 5.3, 5.3, 5.29, 5.26, 5.24, 5.23, 5.2, 5.2, 5.19, 5.18, 5.18, 5.14, 5.14, 5.12, 5.11, 5.11, 5.1, 5.09, 5.09, 5.09, 5.06, 5.05, 5.04, 5.03, 5.03, 5.02, 5.02, 5.02, 5.02, 4.99, 4.98, 4.98, 4.98, 4.97, 4.97, 4.93, 4.91, 4.91, 4.91, 4.9, 4.9, 4.89, 4.87, 4.86, 4.86, 4.86, 4.85, 4.85, 4.85, 4.84, 4.84, 4.84, 4.83, 4.83, 4.82, 4.82, 4.81, 4.8, 4.8, 4.78, 4.76, 4.76, 4.75, 4.73, 4.72, 4.71, 4.7, 4.7, 4.7, 4.69, 4.68, 4.65, 4.65, 4.64, 4.64, 4.63, 4.62, 4.6, 4.59, 4.59, 4.57, 4.57, 4.57, 4.55, 4.55, 4.54, 4.54, 4.53, 4.53, 4.5, 4.48, 4.47, 4.42, 4.41, 4.36, 4.36, 4.35, 4.26, 4.16, 4.13]

ROADMAP_COLUMNS = ["Next", "This week", "Later"]
ROADMAP = [
    ("OCR", [[("Train P and S longer on the GPU", "Gate: P beats S by ≥ 2 points on TS-TR", True)],
             [("Prune D to tiny's size (171 KB)", "Gate: held-out > 64.4%, TS-TR > 32.6%", False),
              ("Quantization-aware training for the pruned model", "Gate: int8 loses ≤ 2 points on every set", False)],
             [("Publish a 298 KB model", "Gate: the gates before it, plus an ESP32-P4 measurement", False)]]),
    ("VoxHOD", [[("Train with many more empty and hard-negative tiles", "Gate: precision > 20.4% at recall ≥ 83%", True)],
                [("Find small people more often", "Gate: ≥ 85% of 8-16 px people, same size", False)],
                [("More varied aerial data", "Gate: WiSARD recall > 61.3% without losing HERIDAL", False)]]),
    ("ASR", [[("Continue v11 on the GPU", "Gate: CER below 40% on both test sets", True)],
             [("Wide teacher, 64k-parameter student", "Gate: the student beats v11 at the same size", False)],
             [("Record the command dataset", "Gate: the command-classifier acceptance threshold", False)]]),
    ("Infra", [[("GPU training in WSL2", "Done 2026-11-07: 155 s per epoch instead of about 50 min", False)],
               [("Feed the GPU faster", "Gate: epoch below 100 s (GPU is ~30% busy)", False)],
               [("Check device output against the PC", "Gate: same results on device and PC", False)]]),
]

D, P2, TINY, P, S = 0, 1, 3, 1, 2  # colour slots follow the model: D blue, P/P2 orange, S aqua, tiny yellow


def lineage() -> None:
    width, height, col_w, x0, bw, bh = 860, 452, 94, 8, 86, 46
    cols = ["Oct 9", "Oct 10", "Oct 25", "Oct 31", "Nov 1", "Nov 2", "Nov 4-6", "Nov 7 CPU", "Nov 7 GPU"]
    rows = {"lstm": 142, "conv": 244, "prune": 340, "tiny": 422}
    nodes = {  # key: (column, row y, name, sub, held-out int8, status)
        "v1009": (0, rows["lstm"], "Oct 9", "LSTM · 665 KB", 32.9, "old"),
        "v1010": (1, rows["lstm"], "Oct 10", "LSTM · 665 KB", 39.5, "old"),
        "v1025": (2, rows["lstm"], "Oct 25", "LSTM · 665 KB", 32.1, "old"),
        "A": (3, rows["lstm"], "A", "LSTM · 665 KB", 48.2, "old"),
        "B": (3, rows["conv"], "B", "conv · 598 KB", 78.4, "old"),
        "D": (4, rows["conv"], "D", "conv · 598 KB", 77.3, "pub"),
        "EF": (5, rows["conv"], "E, F", "598 KB", 77.1, "dead"),
        "P": (7, rows["prune"], "P", "pruned · 298 KB", 71.8, "old"),
        "S": (7, rows["prune"] + 0, "S", "", 65.3, "skip"),
        "P2": (8, rows["prune"], "P2", "298 KB", 72.3, "now"),
        "T": (5, rows["tiny"], "tiny", "171 KB", 64.4, "pub"),
        "K": (6, rows["tiny"], "K, K2", "distilled · 171 KB", 63.6, "dead"),
    }
    del nodes["S"]  # S is the control run; it is shown in the pruning chart instead

    def box(k: str) -> tuple[float, float, float, float]:
        c, y = nodes[k][0], nodes[k][1]
        x = x0 + c * col_w
        return x, y - bh / 2, x + bw / 2, y

    def body(t: dict) -> str:
        out = [header(t, "How the model got here", "Each box: one version, its int8 held-out exact-line accuracy (%). Arrows: what changed.",
                      [("on Hugging Face", D), ("candidate, not published", P2)])]
        lx = 20 + 34 + 6.6 * len("on Hugging Face") + 34 + 6.6 * len("candidate, not published")
        out.append(f'<rect x="{lx:.0f}" y="62" width="12" height="12" rx="3" fill="{t["surface"]}" stroke="{t["muted"]}" stroke-dasharray="3 2"/>'
                   f'<text x="{lx + 18:.0f}" y="72" font-size="12" fill="{t["text"]}">dead end</text>')
        out.append(f'<defs><marker id="a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
                   f'<path d="M0,0L8,4L0,8Z" fill="{t["muted"]}"/></marker>'
                   f'<marker id="b" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
                   f'<path d="M0,0L8,4L0,8Z" fill="{t["series"][1]}"/></marker></defs>')
        for i, c in enumerate(cols):
            out.append(f'<text x="{x0 + i * col_w + bw / 2:.0f}" y="94" font-size="11" text-anchor="middle" fill="{t["muted"]}">{c}</text>')

        def path(d: str, color: str | None = None, dash: bool = False, w: float = 1.5) -> None:
            dash_attr = 'stroke-dasharray="4 3"' if dash else ""
            marker = "b" if color else "a"
            out.append(f'<path d="{d}" fill="none" stroke="{color or t["muted"]}" stroke-width="{w}" '
                       f'{dash_attr} marker-end="url(#{marker})"/>')

        def label(x: float, y: float, text: str, strong: bool = False, anchor: str = "start") -> None:
            out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="10.5" text-anchor="{anchor}" '
                       f'font-weight="{600 if strong else 400}" fill="{t["title"] if strong else t["text"]}">{esc(text)}</text>')

        def straight(a: str, b: str, text: str = "", dash: bool = False) -> None:
            ax, _, _, ay = box(a)
            bx, _, _, _ = box(b)
            path(f"M{ax + bw:.0f},{ay:.0f} H{bx - 2:.0f}", dash=dash)
            if text:
                label((ax + bw + bx) / 2, ay - bh / 2 - 6, text, anchor="middle")

        straight("v1009", "v1010", "+ data")
        straight("v1010", "v1025", "+ 35 epochs")
        ax, ay0, acx, ay = box("v1009")
        path(f"M{acx + 14:.0f},{ay0 + bh:.0f} V{ay + 44:.0f} H{box('A')[2]:.0f} V{box('A')[1] + bh + 2:.0f}")
        label(acx + 20, ay + 40, "from Oct 9 weights, orkungedik data removed")
        path(f"M{acx - 14:.0f},{ay0 + bh:.0f} V{rows['conv']:.0f} H{box('B')[0] - 2:.0f}")
        label(acx - 6, rows["conv"] - 8, "new architecture: conv sequence instead of LSTM", strong=True)
        straight("B", "D", "+ donut documents")
        straight("D", "EF", "longer training", dash=True)
        dx, dy0, dcx, _ = box("D")
        path(f"M{dcx - 12:.0f},{dy0 + bh:.0f} V{rows['prune']:.0f} H{box('P')[0] - 2:.0f}", color=t["series"][1], w=2)
        label(dcx - 4, rows["prune"] - 8, "pruning: 70% of channels kept", strong=True)
        straight("P", "P2", "GPU, +5 epochs")
        path(f"M{dcx + 14:.0f},{dy0 + bh:.0f} V{rows['prune'] - 44:.0f} H{box('K')[2]:.0f} V{box('K')[1] - 2:.0f}", dash=True)
        label(box("K")[2] + 6, rows["prune"] - 30, "D as teacher")
        label(box("T")[0] - 6, rows["tiny"] + 4, "half width, from scratch →", anchor="end")
        for k, (c, y, name, sub, score, status) in nodes.items():
            x, top, _, _ = box(k)
            solid = status in ("pub", "now")
            fill = t["series"][D] if status == "pub" else t["series"][1] if status == "now" else t["surface"]
            ink = t["on_series"] if solid else t["title"]
            ink2 = t["on_series"] if solid else t["text"]
            dash_attr = 'stroke-dasharray="4 3"' if status == "dead" else ""
            out.append(f'<g><title>{esc(name)} · {esc(sub)} · held-out int8 {fmt(score)}%</title>'
                       f'<rect x="{x:.0f}" y="{top:.0f}" width="{bw}" height="{bh}" rx="6" fill="{fill}" '
                       f'stroke="{fill if solid else t["muted"]}" stroke-width="1.5" {dash_attr}/>'
                       f'<text x="{x + 8:.0f}" y="{top + 18:.0f}" font-size="12.5" font-weight="600" fill="{ink}">{esc(name)}</text>'
                       f'<text x="{x + bw - 8:.0f}" y="{top + 18:.0f}" font-size="12.5" font-weight="600" text-anchor="end" fill="{ink}">{fmt(score)}</text>'
                       f'<text x="{x + 8:.0f}" y="{top + 36:.0f}" font-size="10" fill="{ink2}">{esc(sub)}</text></g>')
        return "".join(out)

    write(DOCS / "ocr-lineage", width, height, body)


def frontier() -> None:
    width, height = 860, 420
    left, right, top, bottom = 56, 30, 96, 370
    pw, ph = width - left - right, bottom - top

    def sx(kb: float) -> float:
        return left + pw * kb / 700

    def sy(v: float) -> float:
        return bottom - ph * v / 100

    points = [  # label, KB, held-out int8, kind, label dx, dy, anchor (drawn in this order)
        ("B 78.4", 598, 78.4, "B", -10, -8, "end"),
        ("tiny 64.4", 171, 64.4, "tiny", 10, 18, "start"),
        ("S 65.3 (control, from scratch)", 298, 65.3, "S", 10, 18, "start"),
        ("P2 72.3 (pruned D)", 298, 72.3, "P2", 10, -8, "start"),
        ("D 77.3 (published)", 598, 77.3, "D", -10, 18, "end"),
        ("Oct 10, 39.5", 665, 39.5, "lstm", -10, 4, "end"),
        ("A, 48.2", 665, 48.2, "lstm", -10, 4, "end"),
        ("Oct 9 and Oct 25, ~32", 665, 32.5, "lstm", -10, 4, "end"),
    ]
    kinds = {"D": D, "P2": P2, "S": S, "tiny": TINY}

    def body(t: dict) -> str:
        out = [header(t, "Size against accuracy: the frontier moved left and up",
                      "int8 file size and held-out exact-line accuracy of every measured version.",
                      [("D", D), ("P2", P2), ("S", S), ("tiny", TINY), ("LSTM versions (Oct)", "hollow")])]
        for v in (0, 25, 50, 75, 100):
            out.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{t["grid" if v else "axis"]}"/>'
                       f'<text x="{left - 8}" y="{sy(v) + 4:.1f}" font-size="11" text-anchor="end" fill="{t["muted"]}">{v}%</text>')
        for kb in (0, 100, 200, 300, 400, 500, 600, 700):
            out.append(f'<text x="{sx(kb):.1f}" y="{bottom + 18}" font-size="11" text-anchor="middle" fill="{t["muted"]}">{kb}</text>')
        out.append(f'<text x="{left + pw / 2:.0f}" y="{bottom + 36}" font-size="11.5" text-anchor="middle" fill="{t["text"]}">int8 file size, KB</text>')
        front = [(171, 64.4), (298, 72.3), (598, 78.4)]
        out.append(f'<path d="M{" L".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in front)}" fill="none" stroke="{t["muted"]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        out.append(f'<text x="{sx(450):.0f}" y="{sy(85):.0f}" font-size="11" text-anchor="middle" fill="{t["muted"]}">best so far at each size</text>')
        for name, kb, v, kind, dx, dy, anchor in points:
            x, y = sx(kb), sy(v)
            if kind == "lstm":
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{t["surface"]}" stroke="{t["muted"]}" stroke-width="2"><title>{esc(name)}%, 665 KB</title></circle>')
            elif kind == "B":
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{t["muted"]}" stroke="{t["surface"]}" stroke-width="2"><title>B: 598 KB, 78.4%</title></circle>')
            else:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{t["series"][kinds[kind]]}" stroke="{t["surface"]}" stroke-width="2"><title>{esc(name)}%, {kb} KB</title></circle>')
            strong = kind in ("P2", "D", "tiny")
            out.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" font-size="11.5" text-anchor="{anchor}" '
                       f'font-weight="{600 if strong else 400}" fill="{t["title"] if strong else t["text"]}">{esc(name)}</text>')
        return "".join(out)

    write(DOCS / "ocr-size-frontier", width, height, body)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    lineage()
    frontier()
    dumbbell(DOCS / "ocr-int8-gap", "Converting to int8 used to destroy the model; the conv design fixed it",
             "Held-out exact-line accuracy (12,093 lines) of the float model and of its int8 device file.",
             [{"label": n, "a": f, "b": q, "highlight": arch == "LSTM"} for n, kb, arch, f, q in VERSIONS],
             divider_before=4, divider_label="conv sequence head from here on")
    grouped_bars(DOCS / "ocr-models", "Current models, int8 device files",
                 "Exact-line accuracy. D and tiny are on Hugging Face; P2 (pruned D) is a candidate, not yet published.",
                 ["Held-out lines (12,093)", "TS-TR scene text (2,078)", "Printed documents, donut (2,563)"],
                 [("D · 598 KB", D, [77.3, 56.2, 46.5]), ("P2 · 298 KB", P2, [72.3, 46.0, 33.2]),
                  ("tiny · 171 KB", TINY, [64.4, 32.6, 20.1])])
    grouped_bars(DOCS / "ocr-prune-vs-scratch", "Pruned D against the same model trained from scratch",
                 "Same architecture (298 KB), same data, same 8 epochs; only the starting weights differ. int8 exact-line accuracy.",
                 ["Held-out lines", "TS-TR scene text", "Printed documents, donut"],
                 [("P · pruned from D", P, [71.8, 44.5, 31.7]), ("S · from scratch", S, [65.3, 34.4, 20.6])])
    channel_bars(DOCS / "ocr-pruning-channels", "Which connections pruning cut",
                 "D's 128 sequence channels sorted by importance (BatchNorm |gamma| + weight norm). The pruned model keeps the top 89.",
                 SEQ_SCORES, keep=89)
    roadmap(DOCS / "roadmap", "Roadmap for all three models",
            "Rule: the device model never grows; larger models are only teachers. A step ships only if it passes its gate.",
            ROADMAP_COLUMNS, ROADMAP)
    print("charts written to", DOCS)


if __name__ == "__main__":
    main()
