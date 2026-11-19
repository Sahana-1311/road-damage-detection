# AI-Based Road Damage Detection and Severity Classification (YOLO)

Detects **cracks and potholes** in road images/video with YOLO, then **classifies the severity**
(Low / Medium / High) of every damage and gives the whole image a **road-condition rating**
(Good / Fair / Poor / Critical) with a maintenance recommendation.

## Pipeline
```
road image -> YOLO detector (damage type + box) -> severity module (size + type -> score) -> rating + report
```
Classes (RDD2022 codes): `longitudinal_crack` (D00), `transverse_crack` (D10), `alligator_crack` (D20), `pothole` (D40).

## Severity rule (explainable, see `severity.py`)
`score = 100 * (0.4 * class_weight + 0.6 * min(1, extent / reference))`
- class_weight: pothole 1.0, alligator 0.8, transverse 0.5, longitudinal 0.4
- extent: box **area** (% of image) for potholes/alligator cracks, box **length** for linear cracks
- Low < 35, Medium 35-65, High >= 65
- Image rating: `damage_index = 0.7 * worst_score + 0.3 * coverage_term` -> Good < 25, Fair < 45, Poor < 65, Critical

## Files
| File | Purpose |
|---|---|
| `data.yaml` | dataset paths + 4 class names |
| `convert_rdd_voc.py` | RDD2022 Pascal-VOC XML -> YOLO txt labels |
| `prepare_dataset.py` | split into train/val/test (70/20/10) and validate labels |
| `train.py` / `evaluate.py` | transfer-learning training / precision, recall, mAP -> `metrics.json` |
| `severity.py` | severity scoring + road rating (pure Python, unit-tested) |
| `detectors.py` | `YoloDetector` (real model) and `BaselineDetector` (classical OpenCV, no training) |
| `visualize.py` | severity-coloured boxes, summary panel, CSV log |
| `detect.py` | CLI for image / folder / video / webcam |
| `app.py` | Streamlit web app |
| `make_demo_data.py` | SYNTHETIC images for smoke tests only |
| `tests/test_pipeline.py` | unit tests: `python -m unittest tests/test_pipeline.py` |

## Setup
```
pip install -r requirements.txt
python -m unittest tests/test_pipeline.py      # should say OK
```

## Run the demo RIGHT NOW (no training needed)
```
streamlit run app.py                                   # choose "Classical CV baseline" in the sidebar
python detect.py --source some_road.jpg --baseline
```
The baseline is a heuristic (black-hat filtering + shape rules). It is a fallback and a comparison
baseline, **not** the deep-learning model. On real photos expect many misses/false alarms.

## Real pipeline (this is the project)
1. **Get data** - RDD2022 (Road Damage Detection challenge dataset; search "RDD2022" on Kaggle/Roboflow/figshare),
   or any Roboflow Universe "pothole / road damage" dataset exported in *YOLOv8* format (then skip step 2).
2. **Convert RDD2022 XML -> YOLO labels** (use one country, e.g. India, to keep it small):
   `python convert_rdd_voc.py --src RDD2022/India/train --out raw_data --limit 3000`
3. **Split:** `python prepare_dataset.py --src raw_data --out dataset`
4. **Train** (GPU strongly recommended - Google Colab T4 is enough):
   `python train.py --model yolov8n.pt --epochs 30 --imgsz 640 --batch 16 --device 0`
5. **Evaluate (numbers for the report):** `python evaluate.py --split test`
   Confusion matrix + PR curve are saved in `runs/detect/val*/`.
6. **Run:** `streamlit run app.py` (picks up `runs/road_damage/weights/best.pt` automatically) or
   `python detect.py --source road.jpg`

If your Roboflow dataset has different class names/order, edit `data.yaml`; names like `Pothole`,
`D40`, `alligator crack` are mapped automatically by `severity.normalize_name`.

## Colab quick start
```
!pip install ultralytics
# upload the project folder + dataset zip, then
!unzip -q dataset.zip
!python train.py --device 0 --epochs 30
!python evaluate.py
# download runs/road_damage/weights/best.pt and put it in the same path on your laptop
```

## Smoke test of the whole pipeline (synthetic data - checks the code, NOT accuracy)
```
python make_demo_data.py --out demo_raw --n 80
python prepare_dataset.py --src demo_raw --out dataset
python train.py --epochs 3 --batch 8
```

## Known limitations (say these in the viva)
- Severity is rule-based from box size + damage type, not learned; box area overstates thin diagonal cracks.
- No real-world scale: size is relative to the image, not metres (needs camera calibration).
- Lighting, shadows, wet roads, road markings and manhole covers cause errors.
- Per-image rating; no GPS / cumulative road-segment map.
