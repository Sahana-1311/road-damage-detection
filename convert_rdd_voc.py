"""Convert RDD2022-style Pascal-VOC XML annotations to YOLO .txt labels (4 classes).

RDD2022 ships per-country folders like  <Country>/train/images/*.jpg  and
<Country>/train/annotations/xmls/*.xml  with codes D00 D10 D20 D40 (+ D01 D11 D43 D44 ... which we drop).
Usage:  python convert_rdd_voc.py --src RDD2022/India/train --out raw_data [--limit 3000]
Then:   python prepare_dataset.py --src raw_data --out dataset
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from severity import CLASS_NAMES, RDD_CODES

IMG_EXT = {".jpg", ".jpeg", ".png"}


def convert_xml(xml_path):
    """Return (list of YOLO label lines, (width, height)) for one XML file."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = float(size.findtext("width")), float(size.findtext("height"))
    lines = []
    for obj in root.iter("object"):
        code = (obj.findtext("name") or "").strip().lower()
        if code not in RDD_CODES:
            continue
        cid = CLASS_NAMES.index(RDD_CODES[code])
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines, (w, h)


def convert_folder(src, out, limit=None, seed=42):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT}
    xmls = sorted(src.rglob("*.xml"))
    random.Random(seed).shuffle(xmls)
    if limit:
        xmls = xmls[:limit]
    n = kept = 0
    for x in xmls:
        img = images.get(x.stem)
        if img is None:
            continue
        lines, _ = convert_xml(x)
        shutil.copy2(img, out / img.name)
        (out / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
        kept += bool(lines)
    return n, kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="raw_data")
    ap.add_argument("--limit", type=int, default=None, help="use only N images (faster training)")
    a = ap.parse_args()
    n, kept = convert_folder(a.src, a.out, a.limit)
    print(f"Converted {n} images ({kept} with at least one of the 4 target damage classes) -> {a.out}")
