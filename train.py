"""Train a YOLO detector on the road-damage dataset (transfer learning from COCO weights)."""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data.yaml")
ap.add_argument("--model", default="yolov8n.pt", help="pretrained weights (transfer learning); yolov8s.pt = more accurate")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="", help="'' = auto, '0' = first GPU, 'cpu'")
a = ap.parse_args()

model = YOLO(a.model)
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
    project="runs", name="road_damage", patience=20, seed=42,
    # augmentation: colour jitter, horizontal flip, scale, mosaic (roads vary in light / camera / scale)
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, scale=0.5, mosaic=1.0,
)
print("Best weights: runs/road_damage/weights/best.pt")
