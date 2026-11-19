"""Streamlit front-end:  streamlit run app.py"""
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detectors import BaselineDetector, make_detector
from severity import build_report
from visualize import draw_results

st.set_page_config(page_title="Road Damage Detection", layout="wide")
st.title("AI-Based Road Damage Detection and Severity Classification")
st.caption("Detects cracks and potholes, then grades each damage Low / Medium / High and rates the road.")

DEFAULT_WEIGHTS = "runs/road_damage/weights/best.pt"
have_weights = os.path.exists(DEFAULT_WEIGHTS)
engine = st.sidebar.radio("Detection engine", ["YOLO (trained weights)", "Classical CV baseline"],
                          index=0 if have_weights else 1)
weights = st.sidebar.text_input("Weights path", DEFAULT_WEIGHTS)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.05)


@st.cache_resource
def load(engine_name, path):
    if engine_name.startswith("YOLO"):
        return make_detector(path)
    return BaselineDetector()


detector = load(engine, weights)
if engine.startswith("YOLO") and isinstance(detector, BaselineDetector):
    st.sidebar.warning("Weights file not found - using the classical baseline instead.")
st.sidebar.info(f"Active: {detector.name}")

files = st.file_uploader("Upload road image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
all_rows = []
for up in files or []:
    frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    dets = detector.detect(frame, conf=conf, iou=iou)
    rep = build_report(dets, frame.shape)
    st.subheader(up.name)
    c1, c2 = st.columns([3, 2])
    c1.image(cv2.cvtColor(draw_results(frame, rep), cv2.COLOR_BGR2RGB), use_container_width=True)
    m1, m2, m3 = c2.columns(3)
    m1.metric("Road condition", rep["rating"])
    m2.metric("Damage index", rep["damage_index"])
    m3.metric("Coverage %", rep["coverage_pct"])
    (c2.error if rep["rating"] in ("Poor", "Critical") else c2.success)(rep["advice"])
    if rep["detections"]:
        df = pd.DataFrame(rep["detections"]).drop(columns=["box"])
        c2.dataframe(df, hide_index=True)
        c2.bar_chart(pd.Series(rep["counts"]))
        all_rows += [{"image": up.name, **r} for r in df.to_dict("records")]
    else:
        c2.info("No damage detected.")

if all_rows:
    st.download_button("Download all detections (CSV)", pd.DataFrame(all_rows).to_csv(index=False),
                       "road_damage_detections.csv", "text/csv")
