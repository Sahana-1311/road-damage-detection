"""Split a flat folder of images + YOLO .txt labels into train/val/test and
check every label file (5 values per line, valid class id, coords in 0-1).
Images with an empty .txt are kept as 'no damage' background images."""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def check_label(path, num_classes):
    problems = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{path.name}:{i} expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            problems.append(f"{path.name}:{i} non-numeric value")
            continue
        if not 0 <= cls < num_classes:
            problems.append(f"{path.name}:{i} class id {cls} out of range")
        if any(v < 0 or v > 1 for v in vals):
            problems.append(f"{path.name}:{i} coordinate outside 0-1")
    return problems


def split_dataset(src, out, ratios=(0.7, 0.2, 0.1), seed=42, num_classes=4):
    src, out = Path(src), Path(out)
    pairs = [(img, img.with_suffix(".txt")) for img in sorted(src.iterdir())
             if img.suffix.lower() in IMG_EXT and img.with_suffix(".txt").exists()]
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    parts = {"train": pairs[:a], "val": pairs[a:b], "test": pairs[b:]}
    problems = []
    for split, items in parts.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lab in items:
            problems += check_label(lab, num_classes)
            shutil.copy2(img, out / "images" / split / img.name)
            shutil.copy2(lab, out / "labels" / split / lab.name)
    return {k: len(v) for k, v in parts.items()}, problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="folder with images and same-name .txt labels")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    counts, problems = split_dataset(a.src, a.out, seed=a.seed, num_classes=a.classes)
    print("Split sizes:", counts)
    print(f"{len(problems)} label problem(s)")
    for p in problems[:20]:
        print("  ", p)
