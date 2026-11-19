"""Drawing + CSV logging for results (cv2 / numpy only)."""
import csv
import os
from datetime import datetime

import cv2

from severity import CLASS_NAMES, LEVELS

COLORS = {"Low": (0, 200, 0), "Medium": (0, 165, 255), "High": (0, 0, 255)}   # BGR
RATING_COLORS = {"Good": (0, 200, 0), "Fair": (0, 215, 255), "Poor": (0, 140, 255), "Critical": (0, 0, 255)}


def draw_results(frame, report):
    """Boxes coloured by severity + a summary panel (road rating, damage index, counts)."""
    out = frame.copy()
    th = max(2, round(min(out.shape[:2]) / 250))
    for it in report["detections"]:
        x1, y1, x2, y2 = it["box"]
        col = COLORS[it["severity"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        label = f'{it["damage"]} | {it["severity"]} | {it["confidence"]:.2f}'
        (tw, tH), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - tH - 6)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + tH + 6), col, -1)
        cv2.putText(out, label, (x1 + 3, top + tH + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    lines = [(f'Road: {report["rating"]}  (index {report["damage_index"]})', RATING_COLORS[report["rating"]])]
    lines += [(f"{n}: {c}", (255, 255, 255)) for n, c in report["counts"].items() if c]
    if len(lines) == 1:
        lines.append(("No damage detected", (255, 255, 255)))
    panel_w = max(260, max(len(t) for t, _ in lines) * 10)
    panel_h = 12 + 26 * len(lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (30, 30, 30), -1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    for i, (text, col) in enumerate(lines):
        cv2.putText(out, text, (14, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return out


def log_report(csv_path, source, report):
    """Append one timestamped summary row per image/frame."""
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "source", "rating", "damage_index", "coverage_pct",
                        *CLASS_NAMES, *[f"{l.lower()}_severity" for l in LEVELS]])
        w.writerow([datetime.now().isoformat(timespec="seconds"), source, report["rating"],
                    report["damage_index"], report["coverage_pct"],
                    *report["counts"].values(), *report["levels"].values()])
