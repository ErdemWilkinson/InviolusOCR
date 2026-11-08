"""Compare the device's argmax output (serial.log) with the PC TFLite reference and the ground truth.

    python compare.py serial.log reference.json ../../artifacts_D_1101/alphabet.json
"""
import json, re, sys

log, ref_path, alphabet_path = sys.argv[1:4]
ref = json.load(open(ref_path, encoding="utf-8")); n = len(ref["texts"])
alphabet = json.load(open(alphabet_path, encoding="utf-8"))["alphabet"]
blank = len(alphabet) + 1


def decode(ids):
    out, prev = [], None
    for i in ids:
        if i != prev and 0 < i < blank:
            out.append(alphabet[i - 1])
        prev = i
    return "".join(out)


device = {}
for line in open(log, encoding="utf-8", errors="replace"):
    m = re.match(r"IDS model=(\S+) i=(\d+) ([0-9a-f]+)", line.strip())
    if m:
        device.setdefault(m[1], {})[int(m[2])] = [int(m[3][i:i + 2], 16) for i in range(0, len(m[3]), 2)]
truth, result = ref["texts"], {"lines": n}
for name in [k for k in ref if k != "texts"]:
    pc = [decode(x) for x in ref[name]]
    dev = [decode(device[name][i]) for i in range(n)]
    result[name] = {"same_text": sum(a == b for a, b in zip(pc, dev)),
                    "pc_exact": sum(a == t for a, t in zip(pc, truth)),
                    "device_exact": sum(a == t for a, t in zip(dev, truth)),
                    "pc_right_device_wrong": sum(a == t != b for a, b, t in zip(pc, dev, truth)),
                    "pc_wrong_device_right": sum(b == t != a for a, b, t in zip(pc, dev, truth))}
print(json.dumps(result, indent=1))
