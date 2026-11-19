# Viva prep - likely questions (answer in your own words)

1. **Problem statement?** Manual road inspection is slow and subjective. We detect cracks/potholes automatically and grade how bad they are so repairs can be prioritised.
2. **Why YOLO?** Single-stage detector: one forward pass gives boxes + classes, fast enough for dashcam video; two-stage (Faster R-CNN) is usually slower.
3. **Detection vs classification vs segmentation?** Detection = what + where (box). Severity needs size, so boxes are enough here; segmentation would give exact crack area/length (future scope).
4. **Dataset?** RDD2022 (or the one you used): state number of images, country, the 4 classes and the train/val/test split (70/20/10).
5. **Transfer learning?** Start from COCO-pretrained YOLO weights, fine-tune on road damage -> good results with little data/time.
6. **How is severity decided?** Rule-based: score = 100*(0.4*class weight + 0.6*normalised size). Pothole weighs most; size = area for potholes/alligator, length for linear cracks. Low <35, Medium 35-65, High >=65.
7. **Why not train a severity classifier?** RDD has no severity labels. A rule is transparent and adjustable; with severity-labelled data a CNN classifier could replace it (future scope).
8. **Road rating?** damage_index = 0.7*worst damage + 0.3*coverage -> Good/Fair/Poor/Critical with a maintenance recommendation.
9. **Metrics?** Precision = TP/(TP+FP); recall = TP/(TP+FN); mAP@0.5 and mAP@0.5:0.95. Quote YOUR numbers from `metrics.json` and say which class is weakest.
10. **NMS / IoU / confidence threshold?** NMS removes duplicate boxes using IoU (0.45); higher confidence threshold = fewer false alarms, more misses (0.25 default).
11. **Augmentation?** HSV jitter, flip, scale, mosaic - roads vary in light, camera and distance.
12. **What is the classical baseline in the app?** OpenCV black-hat + shape rules, no learning. It is a fallback and a comparison to show why deep learning is needed.
13. **Limitations?** Shadows/wet roads/markings, size is relative to image not metres, cracks hard to see at distance, box area overstates diagonal cracks.
14. **Future scope?** Segmentation for exact area, GPS-tagged road-segment map, camera calibration for real size, edge deployment (Jetson/phone), severity learned from labelled data.
