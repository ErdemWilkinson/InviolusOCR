"""Turkish recognition 200k v4 (turgutguvercin/donut, CC BY-SA 4.0) -> line crops for train.py.

Only rows from the dataset's own *recommended* lists are used (its quality
filter drops deliberately ambiguous crops). Rows must also fit this model:
1-24 characters, all inside alphabet.json. Images are copied as they are;
train.py resizes them to 160x32.

    python scripts/prepare_donut.py D:/datasets/raw/donut-turkish-ocr-200k-v4 D:/ai-egitim-veri/ocr_new

Writes <out>/donut_lines/*.jpg, <out>/donut_labels.csv (train split) and
<out>/donut_test_labels.csv (test split, evaluation only).
"""
from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "turkish-recognition-200k-v4/"
MAX_TEXT_LENGTH = 24


def read_list(zips: list[Path], name: str) -> list[tuple[str, str]]:
    for path in zips:
        with zipfile.ZipFile(path) as archive:
            if PREFIX + name in archive.namelist():
                lines = archive.read(PREFIX + name).decode("utf-8").splitlines()
                return [tuple(line.split("\t", 1)) for line in lines if "\t" in line]
    raise SystemExit(f"{name} not found in the archives")


def main() -> None:
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    zips = sorted(source.glob("*.zip"))
    alphabet = set(json.loads((ROOT / "artifacts" / "alphabet.json").read_text(encoding="utf-8"))["alphabet"])
    wanted: dict[str, tuple[str, str]] = {}  # image path -> (split, text)
    for split in ("train", "test"):
        rows = read_list(zips, f"{split}-recommended.txt")
        kept = [(image, text) for image, text in rows
                if 1 <= len(text) <= MAX_TEXT_LENGTH and set(text) <= alphabet]
        print(f"{split}: {len(rows)} recommended rows, {len(kept)} fit 1-{MAX_TEXT_LENGTH} chars and the alphabet")
        for image, text in kept:
            wanted[image] = (split, text)

    images = out / "donut_lines"
    images.mkdir(parents=True, exist_ok=True)
    written = {"train": [], "test": []}
    for path in zips:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                image = member[len(PREFIX):]
                if image in wanted:
                    split, text = wanted.pop(image)
                    name = Path(image).name
                    (images / name).write_bytes(archive.read(member))
                    # group = source folder of 1000 images, so neighbours stay together
                    written[split].append({"image": f"donut_lines/{name}", "text": text,
                                           "group": f"donut_{Path(image).parent.name}"})
    for split, file in (("train", "donut_labels.csv"), ("test", "donut_test_labels.csv")):
        with (out / file).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "text", "group"])
            writer.writeheader()
            writer.writerows(written[split])
        print(f"{file}: {len(written[split])} rows")
    if wanted:
        print(f"{len(wanted)} listed images were not found in the archives")


if __name__ == "__main__":
    main()
