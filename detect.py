"""Run road-damage detection + severity classification on an image, folder, video or webcam.

  python detect.py --source road.jpg                          # uses runs/road_damage/weights/best.pt if present
  python detect.py --source road.jpg --baseline               # classical CV, no training needed
  python detect.py --source video.mp4 --weights best.pt
Outputs (annotated media + road_damage_log.csv + detections.csv) go to --out.
"""
import argparse
import csv
import glob
import os

import cv2

from detectors import BaselineDetector, make_detector
from severity import build_report
from visualize import draw_results, log_report

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def process(detector, frame, conf, iou):
    dets = detector.detect(frame, conf=conf, iou=iou)
    report = build_report(dets, frame.shape)
    return draw_results(frame, report), report


def save_detections(path, source, report):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["source", "damage", "confidence", "severity", "severity_score", "extent_type", "extent_pct", "box"])
        for it in report["detections"]:
            w.writerow([source, it["damage"], it["confidence"], it["severity"], it["severity_score"],
                        it["extent_type"], it["extent_pct"], it["box"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image / folder / video / webcam index (0)")
    ap.add_argument("--weights", default="runs/road_damage/weights/best.pt")
    ap.add_argument("--baseline", action="store_true", help="force the classical OpenCV detector")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="")
    ap.add_argument("--log-every", type=int, default=30, help="video: log a row every N frames")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    detector = BaselineDetector() if a.baseline else make_detector(a.weights, a.imgsz, a.device)
    print("Detector:", detector.name)
    log_path, det_path = os.path.join(a.out, "road_damage_log.csv"), os.path.join(a.out, "detections.csv")

    src = a.source
    if src.isdigit() or src.lower().endswith(VIDEO_EXT):
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(os.path.join(a.out, "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        i = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            out, rep = process(detector, frame, a.conf, a.iou)
            writer.write(out)
            if i % a.log_every == 0:
                log_report(log_path, f"{os.path.basename(src)}#frame{i}", rep)
                print(f"frame {i}: {rep['rating']} (index {rep['damage_index']}) {rep['counts']}")
            if a.show:
                cv2.imshow("Road damage", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            i += 1
        cap.release()
        writer.release()
    else:
        files = (sorted(f for f in glob.glob(os.path.join(src, "*")) if f.lower().endswith(IMAGE_EXT))
                 if os.path.isdir(src) else [src])
        for f in files:
            frame = cv2.imread(f)
            if frame is None:
                print("Could not read", f)
                continue
            out, rep = process(detector, frame, a.conf, a.iou)
            name = os.path.basename(f)
            cv2.imwrite(os.path.join(a.out, "det_" + name), out)
            log_report(log_path, name, rep)
            save_detections(det_path, name, rep)
            print(f"{name}: {rep['rating']} (index {rep['damage_index']})  {rep['counts']}  -> {rep['advice']}")
    print("Results saved in", a.out)


if __name__ == "__main__":
    main()
