"""Generate SYNTHETIC road images + YOLO labels for smoke-testing the pipeline.

These are NOT real roads: use them only to check that prepare_dataset -> train -> detect
runs end to end. Train on a real dataset (RDD2022 / Roboflow) for real results.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from severity import CLASS_NAMES


def asphalt(rng, h, w):
    base = rng.integers(95, 130)
    img = np.full((h, w), base, np.float32) + rng.normal(0, 9, (h, w)).astype(np.float32)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    big = cv2.resize(rng.normal(0, 10, (h // 40 + 2, w // 40 + 2)).astype(np.float32), (w, h))
    img = np.clip(img + big, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def crack_path(rng, p0, p1, steps=14, jitter=6):
    pts = []
    for t in np.linspace(0, 1, steps):
        x = p0[0] + (p1[0] - p0[0]) * t + rng.normal(0, jitter)
        y = p0[1] + (p1[1] - p0[1]) * t + rng.normal(0, jitter)
        pts.append((int(x), int(y)))
    return np.array(pts, np.int32)


def draw_poly(img, pts, thick):
    cv2.polylines(img, [pts], False, (25, 25, 25), thick, cv2.LINE_AA)


def add_damage(rng, img, cls):
    h, w = img.shape[:2]
    if cls == 0:      # longitudinal: runs roughly vertical
        x = int(rng.integers(w // 5, 4 * w // 5)); L = int(rng.integers(h // 3, int(h * .85)))
        y0 = int(rng.integers(0, h - L)); pts = crack_path(rng, (x, y0), (x + rng.integers(-30, 30), y0 + L))
        draw_poly(img, pts, int(rng.integers(2, 4)))
        return pts.min(0), pts.max(0)
    if cls == 1:      # transverse: runs roughly horizontal
        y = int(rng.integers(h // 5, 4 * h // 5)); L = int(rng.integers(w // 3, int(w * .85)))
        x0 = int(rng.integers(0, w - L)); pts = crack_path(rng, (x0, y), (x0 + L, y + rng.integers(-30, 30)))
        draw_poly(img, pts, int(rng.integers(2, 4)))
        return pts.min(0), pts.max(0)
    if cls == 2:      # alligator: dense network inside a patch
        pw, ph = int(rng.integers(w // 5, w // 3)), int(rng.integers(h // 5, h // 3))
        x0, y0 = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
        for _ in range(int(rng.integers(14, 22))):
            a = (x0 + rng.integers(0, pw), y0 + rng.integers(0, ph))
            b = (x0 + rng.integers(0, pw), y0 + rng.integers(0, ph))
            draw_poly(img, crack_path(rng, a, b, 5, 3), 2)
        return np.array([x0, y0]), np.array([x0 + pw, y0 + ph])
    # pothole: dark irregular ellipse with lighter rim
    ax, ay = int(rng.integers(w // 16, w // 7)), int(rng.integers(h // 16, h // 7))
    cx, cy = int(rng.integers(ax + 5, w - ax - 5)), int(rng.integers(ay + 5, h - ay - 5))
    ang = int(rng.integers(0, 180))
    cv2.ellipse(img, (cx, cy), (ax + 4, ay + 4), ang, 0, 360, (150, 150, 150), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (ax, ay), ang, 0, 360, (35, 35, 35), -1, cv2.LINE_AA)
    r = max(ax, ay) + 4
    return np.array([cx - r, cy - r]), np.array([cx + r, cy + r])


def make(out, n, seed, size=(480, 640)):
    rng = np.random.default_rng(seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    h, w = size
    for i in range(n):
        img = asphalt(rng, h, w)
        lines = []
        for _ in range(int(rng.integers(0, 4))):          # 0-3 damages per image (0 = background image)
            cls = int(rng.integers(0, 4))
            lo, hi = add_damage(rng, img, cls)
            x1, y1 = np.clip(lo - 3, 0, [w - 1, h - 1]); x2, y2 = np.clip(hi + 3, 0, [w - 1, h - 1])
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            lines.append(f"{cls} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        img = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imwrite(str(out / f"road_{i:04d}.jpg"), img)
        (out / f"road_{i:04d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote {n} synthetic images + labels to {out}  (classes: {CLASS_NAMES})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="demo_raw")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.out, a.n, a.seed)
