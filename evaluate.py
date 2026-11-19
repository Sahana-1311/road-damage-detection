"""Evaluate trained weights: overall and per-class precision / recall / mAP -> metrics.json"""
import argparse
import json

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/road_damage/weights/best.pt")
ap.add_argument("--data", default="data.yaml")
ap.add_argument("--split", default="test", choices=["val", "test"])
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()

model = YOLO(a.weights)
m = model.val(data=a.data, split=a.split, imgsz=a.imgsz, plots=True)   # saves confusion matrix, PR curve

report = {"split": a.split, "precision": float(m.box.mp), "recall": float(m.box.mr),
          "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "per_class": {}}
for k, ci in enumerate(m.box.ap_class_index):
    report["per_class"][model.names[int(ci)]] = {
        "precision": float(m.box.p[k]), "recall": float(m.box.r[k]),
        "mAP50": float(m.box.ap50[k]), "mAP50-95": float(m.box.ap[k])}
print(json.dumps(report, indent=2))
with open("metrics.json", "w") as f:
    json.dump(report, f, indent=2)
print("Saved metrics.json (confusion matrix / PR curve are in the 'runs/detect/val*' folder)")
