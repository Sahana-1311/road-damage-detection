"""Severity classification logic for road damage (pure Python + numpy, no YOLO/torch).

Pipeline idea: the detector says WHAT damage is present and WHERE (class + box).
This module decides HOW BAD it is with a transparent, explainable rule:

    score (0-100) = 100 * ( 0.4 * class_weight  +  0.6 * min(1, extent / reference) )

* class_weight  - inherent danger of the damage type (pothole > alligator > transverse > longitudinal)
* extent        - how big the damage is, measured from the bounding box:
                    potholes / alligator cracks -> area as a fraction of the image
                    longitudinal / transverse cracks -> length as a fraction of the image
* reference     - the extent at which that damage type counts as "fully severe"

Score -> level:  < 35 Low,  35-65 Medium,  >= 65 High.
Whole-image road condition combines the worst damage with total damaged coverage.
"""
from dataclasses import dataclass

import numpy as np

CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]

RDD_CODES = {"d00": "longitudinal_crack", "d10": "transverse_crack",
             "d20": "alligator_crack", "d40": "pothole"}

CLASS_WEIGHT = {"longitudinal_crack": 0.4, "transverse_crack": 0.5,
                "alligator_crack": 0.8, "pothole": 1.0}

# (extent type, reference extent at which damage is "fully severe")
EXTENT_RULE = {"longitudinal_crack": ("length", 0.60), "transverse_crack": ("length", 0.60),
               "alligator_crack": ("area", 0.15), "pothole": ("area", 0.04)}

LOW_MAX, MEDIUM_MAX = 35.0, 65.0
LEVELS = ["Low", "Medium", "High"]

ADVICE = {
    "Good": "No action needed - routine monitoring.",
    "Fair": "Schedule preventive maintenance (crack sealing).",
    "Poor": "Plan repair soon (patching / resurfacing).",
    "Critical": "Immediate repair required - hazard to traffic.",
}


@dataclass
class Detection:
    cls_id: int
    conf: float
    box: tuple  # (x1, y1, x2, y2) in pixels

    @property
    def name(self):
        return CLASS_NAMES[self.cls_id]


def normalize_name(raw):
    """Map any dataset label ('D40', 'Pothole', 'alligator crack'...) to one of CLASS_NAMES, else None."""
    s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if s in CLASS_NAMES:
        return s
    if s in RDD_CODES:
        return RDD_CODES[s]
    for key, val in (("pothole", "pothole"), ("alligator", "alligator_crack"),
                     ("transverse", "transverse_crack"), ("longitudinal", "longitudinal_crack"),
                     ("crack", "longitudinal_crack")):
        if key in s:
            return val
    return None


def extent_of(det, shape):
    """Return (extent_type, extent_value) for one detection given image shape (h, w, ...)."""
    h, w = shape[:2]
    x1, y1, x2, y2 = det.box
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    kind = EXTENT_RULE[det.name][0]
    if kind == "area":
        return kind, float(min(1.0, (bw * bh) / float(w * h)))
    return kind, float(min(1.0, max(bw / w, bh / h)))


def severity_level(score):
    if score < LOW_MAX:
        return "Low"
    if score < MEDIUM_MAX:
        return "Medium"
    return "High"


def assess(det, shape):
    """Score one detection. Returns a plain dict (easy to show in a table / save to CSV)."""
    kind, ext = extent_of(det, shape)
    ref = EXTENT_RULE[det.name][1]
    score = 100.0 * (0.4 * CLASS_WEIGHT[det.name] + 0.6 * min(1.0, ext / ref))
    return {"damage": det.name, "confidence": round(float(det.conf), 3),
            "extent_type": kind, "extent_pct": round(ext * 100, 2),
            "severity_score": round(score, 1), "severity": severity_level(score),
            "box": tuple(int(v) for v in det.box)}


def coverage_fraction(dets, shape):
    """Fraction of the image covered by the union of all damage boxes."""
    h, w = shape[:2]
    mask = np.zeros((h, w), bool)
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return float(mask.mean())


def road_condition(damage_index):
    if damage_index < 25:
        return "Good"
    if damage_index < 45:
        return "Fair"
    if damage_index < 65:
        return "Poor"
    return "Critical"


def build_report(dets, shape):
    """Assess every detection and summarise the whole image."""
    items = [assess(d, shape) for d in dets]
    counts = {n: 0 for n in CLASS_NAMES}
    levels = {l: 0 for l in LEVELS}
    for it in items:
        counts[it["damage"]] += 1
        levels[it["severity"]] += 1
    coverage = coverage_fraction(dets, shape) if dets else 0.0
    worst = max((it["severity_score"] for it in items), default=0.0)
    index = 0.7 * worst + 0.3 * min(100.0, coverage * 100 * 4)   # 25% coverage saturates the 2nd term
    index = round(min(100.0, index), 1)
    rating = road_condition(index)
    return {"detections": items, "counts": counts, "levels": levels,
            "coverage_pct": round(coverage * 100, 2), "damage_index": index,
            "rating": rating, "advice": ADVICE[rating]}
