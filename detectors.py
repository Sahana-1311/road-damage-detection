"""Two interchangeable detectors that return a list of severity.Detection.

* YoloDetector      - the real deep-learning model (needs trained weights + ultralytics)
* BaselineDetector  - classical OpenCV heuristics, NO training needed. Rough, but it lets the
                      app run end to end without weights and serves as a baseline to compare YOLO against.
"""
import os

import cv2
import numpy as np

from severity import CLASS_NAMES, Detection, normalize_name


class YoloDetector:
    name = "YOLO (trained weights)"

    def __init__(self, weights, imgsz=640, device=""):
        from ultralytics import YOLO  # imported lazily so the rest works without it
        self.model = YOLO(weights)
        self.imgsz, self.device = imgsz, device
        self.map = {}
        for i, raw in self.model.names.items():
            norm = normalize_name(raw)
            if norm is not None:
                self.map[int(i)] = CLASS_NAMES.index(norm)

    def detect(self, frame, conf=0.25, iou=0.45):
        res = self.model.predict(frame, conf=conf, iou=iou, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        dets = []
        if res.boxes is None:
            return dets
        for b in res.boxes:
            cid = self.map.get(int(b.cls))
            if cid is None:
                continue
            dets.append(Detection(cid, float(b.conf), tuple(float(v) for v in b.xyxy[0].tolist())))
        return dets


class BaselineDetector:
    name = "Classical CV baseline (no training)"

    WORK = 640          # process at <= 640 px on the long side

    def detect(self, frame, conf=0.25, iou=0.45):
        h, w = frame.shape[:2]
        s = min(1.0, self.WORK / max(h, w))
        small = cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.GaussianBlur(gray, (5, 5), 0))
        pot, pot_mask = self._potholes(gray)
        cracks = self._cracks(gray, pot_mask)
        out = []
        for cid, c, (x1, y1, x2, y2) in pot + cracks:
            if c >= conf:
                out.append(Detection(cid, c, (x1 / s, y1 / s, x2 / s, y2 / s)))
        return out

    # --- potholes: big, compact, clearly darker than the surrounding road -----------------
    def _potholes(self, gray):
        h, w = gray.shape
        # background brightness from a heavily smoothed (1/4 scale) copy so big potholes don't pollute it
        bg = cv2.resize(cv2.medianBlur(cv2.resize(gray, None, fx=.25, fy=.25), 41), (w, h))
        dark = ((bg.astype(np.int16) - gray.astype(np.int16)) > 35).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found, mask = [], np.zeros_like(dark)
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, bw, bh = cv2.boundingRect(c)
            hull = cv2.contourArea(cv2.convexHull(c)) or 1
            if area / (h * w) < 0.004 or area / hull < 0.75 or not 0.3 < bw / bh < 3.3:
                continue
            if area / (bw * bh) < 0.45:
                continue
            m = np.zeros_like(dark); cv2.drawContours(m, [c], -1, 255, -1)
            contrast = float(bg[m > 0].mean() - gray[m > 0].mean())
            found.append((3, float(min(0.95, 0.4 + contrast / 120)), (x, y, x + bw, y + bh)))
            cv2.drawContours(mask, [c], -1, 255, -1)
        return found, cv2.dilate(mask, np.ones((9, 9), np.uint8))

    # --- cracks: thin dark structures (black-hat); orientation-selective line opening separates
    #     "along the road" (vertical) from "across the road" (horizontal) cracks even where they cross;
    #     dense clusters of crack pixels are reported as alligator (network) cracking ---------------
    def _cracks(self, gray, pot_mask):
        h, w = gray.shape
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        mask = (bh > 22).astype(np.uint8) * 255
        mask[pot_mask > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        thick = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        out = []

        # 1) alligator: dense crack regions
        dens = cv2.blur((thick > 0).astype(np.float32), (41, 41))
        reg = (dens > 0.22).astype(np.uint8) * 255
        reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(reg, connectivity=8)
        allig = np.zeros_like(mask)
        for i in range(1, n):
            x, y, bw, bhh, area = st[i]
            # a network patch is dense in BOTH directions; a single long crack is only a thin strip
            if area >= 0.02 * h * w and min(bw, bhh) >= 0.15 * min(h, w):
                c = float(min(0.9, 0.35 + dens[lab == i].mean()))
                out.append((2, c, (x, y, x + bw, y + bhh)))
                allig[lab == i] = 255
        thick[cv2.dilate(allig, np.ones((15, 15), np.uint8)) > 0] = 0

        # 2) long straight-ish cracks, by direction
        k = max(15, int(0.06 * max(h, w)))
        for cid, kern in ((0, np.ones((k, 1), np.uint8)), (1, np.ones((1, k), np.uint8))):
            op = cv2.morphologyEx(cv2.dilate(thick, np.ones((3, 3), np.uint8)), cv2.MORPH_OPEN, kern)
            n, lab, st, _ = cv2.connectedComponentsWithStats(op, connectivity=8)
            for i in range(1, n):
                x, y, bw, bhh, area = st[i]
                if max(bw, bhh) < 0.10 * max(h, w):          # too short to be a crack
                    continue
                if (cid == 0 and bhh < 1.5 * bw) or (cid == 1 and bw < 1.5 * bhh):
                    continue                                   # wrong shape for this direction
                c = float(min(0.9, 0.3 + bh[lab == i].mean() / 80))
                out.append((cid, c, (x, y, x + bw, y + bhh)))
        return self._merge_fragments(out, margin=int(0.06 * max(h, w)))

    @staticmethod
    def _merge_fragments(dets, margin):
        """A wiggly crack breaks into several pieces; join same-class cracks whose boxes nearly touch."""
        dets = list(dets)
        changed = True
        while changed:
            changed = False
            for i in range(len(dets)):
                for j in range(i + 1, len(dets)):
                    a, b = dets[i], dets[j]
                    if a[0] != b[0] or a[0] == 2:
                        continue
                    ax1, ay1, ax2, ay2 = a[2]; bx1, by1, bx2, by2 = b[2]
                    if ax1 - margin < bx2 and bx1 - margin < ax2 and ay1 - margin < by2 and by1 - margin < ay2:
                        box = (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))
                        dets[i] = (a[0], max(a[1], b[1]), box)
                        del dets[j]
                        changed = True
                        break
                if changed:
                    break
        return dets


def make_detector(weights=None, imgsz=640, device=""):
    """YOLO if a weights file exists, otherwise the classical baseline."""
    if weights and os.path.exists(weights):
        return YoloDetector(weights, imgsz, device)
    return BaselineDetector()
