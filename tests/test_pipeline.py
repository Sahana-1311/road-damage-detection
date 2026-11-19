import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convert_rdd_voc import convert_folder, convert_xml
from detectors import BaselineDetector
from make_demo_data import make
from prepare_dataset import check_label, split_dataset
from severity import Detection, assess, build_report, normalize_name, severity_level
from visualize import draw_results, log_report

SHAPE = (480, 640, 3)


class SeverityTests(unittest.TestCase):
    def test_name_normalisation(self):
        self.assertEqual(normalize_name("D40"), "pothole")
        self.assertEqual(normalize_name("Alligator Crack"), "alligator_crack")
        self.assertEqual(normalize_name("D10"), "transverse_crack")
        self.assertEqual(normalize_name("crack"), "longitudinal_crack")
        self.assertIsNone(normalize_name("car"))

    def test_bigger_pothole_is_more_severe(self):
        small = assess(Detection(3, .9, (100, 100, 130, 130)), SHAPE)
        big = assess(Detection(3, .9, (100, 100, 300, 300)), SHAPE)
        self.assertLess(small["severity_score"], big["severity_score"])
        self.assertEqual(big["severity"], "High")

    def test_pothole_worse_than_crack_of_same_box(self):
        box = (100, 100, 200, 200)
        self.assertGreater(assess(Detection(3, .9, box), SHAPE)["severity_score"],
                           assess(Detection(0, .9, box), SHAPE)["severity_score"])

    def test_levels(self):
        self.assertEqual([severity_level(s) for s in (0, 34.9, 35, 64.9, 65, 100)],
                         ["Low", "Low", "Medium", "Medium", "High", "High"])

    def test_empty_report_is_good(self):
        r = build_report([], SHAPE)
        self.assertEqual((r["rating"], r["damage_index"], r["coverage_pct"]), ("Good", 0.0, 0.0))

    def test_report_counts_and_rating(self):
        dets = [Detection(3, .9, (50, 50, 350, 350)), Detection(0, .8, (10, 10, 20, 200))]
        r = build_report(dets, SHAPE)
        self.assertEqual(r["counts"]["pothole"], 1)
        self.assertEqual(sum(r["levels"].values()), 2)
        self.assertEqual(r["rating"], "Critical")


class VisualizeTests(unittest.TestCase):
    def test_draw_and_log(self):
        img = np.full(SHAPE, 120, np.uint8)
        rep = build_report([Detection(3, .9, (50, 50, 200, 200))], SHAPE)
        out = draw_results(img, rep)
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "log.csv")
            log_report(p, "a.jpg", rep); log_report(p, "b.jpg", build_report([], SHAPE))
            rows = list(csv.reader(open(p)))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0][:3], ["timestamp", "source", "rating"])


class DatasetTests(unittest.TestCase):
    def test_split_and_label_check(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d, "src"), Path(d, "out")
            make(src, 10, seed=1, size=(96, 128))
            counts, problems = split_dataset(src, out)
            self.assertEqual(counts, {"train": 7, "val": 2, "test": 1})
            self.assertEqual(problems, [])

    def test_bad_labels_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "x.txt")
            p.write_text("9 0.5 0.5 0.2 0.2\n0 1.5 0.5 0.2 0.2\n0 0.5 0.5\n")
            self.assertEqual(len(check_label(p, 4)), 3)

    def test_rdd_voc_conversion(self):
        xml = """<annotation><size><width>100</width><height>50</height></size>
        <object><name>D40</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>25</ymax></bndbox></object>
        <object><name>D00</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object>
        <object><name>D44</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
        </annotation>"""
        with tempfile.TemporaryDirectory() as d:
            x = Path(d, "a.xml"); x.write_text(xml)
            lines, size = convert_xml(x)
            self.assertEqual(size, (100.0, 50.0))
            self.assertEqual(len(lines), 2)                     # D44 dropped
            self.assertTrue(lines[0].startswith("3 0.200000 0.300000 0.200000 0.400000"))
            cv2.imwrite(str(Path(d, "a.jpg")), np.zeros((50, 100, 3), np.uint8))
            n, kept = convert_folder(d, Path(d, "out"))
            self.assertEqual((n, kept), (1, 1))


class BaselineTests(unittest.TestCase):
    def test_finds_dark_pothole_and_ignores_clean_road(self):
        rng = np.random.default_rng(0)
        road = np.clip(rng.normal(115, 6, (480, 640)), 0, 255).astype(np.uint8)
        road = cv2.cvtColor(cv2.GaussianBlur(road, (3, 3), 0), cv2.COLOR_GRAY2BGR)
        det = BaselineDetector()
        self.assertEqual(det.detect(road), [])
        cv2.ellipse(road, (320, 240), (60, 45), 0, 0, 360, (30, 30, 30), -1)
        found = det.detect(road)
        self.assertTrue(any(d.name == "pothole" and abs((d.box[0] + d.box[2]) / 2 - 320) < 15 for d in found))


if __name__ == "__main__":
    unittest.main()
