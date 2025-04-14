import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
from datetime import datetime
from paddleocr import PaddleOCR
import logging
import tkinter as tk
from tkinter import messagebox, ttk
from utils import show_phone_input_gui

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Hardcoded Configuration
CONFIG = {
    "model_path": "models/yolov8n.pt",
    "detection_confidence": 0.4,
    "violation_duration": 5,  # Detect after 5 seconds
    "stationary_threshold": 5,
    "frame_skip": 5,
    "save_output": False,
    "ocr_params": {
        "use_angle_cls": True,
        "lang": "en",
        "show_log": False
    }
}


def normalize_plate_text(plate_text):
    """Normalize license plate text by removing hyphens, converting to uppercase, and applying specific character replacements."""
    if not plate_text:
        return None
    normalized = plate_text.replace("-", "").replace(" ", "").upper()
    if normalized and normalized[0] == 'O':
        normalized = 'D' + normalized[1:]
    count = 0
    for i in range(len(normalized)):
        if normalized[i] == '5':
            count += 1
            if count == 2:
                normalized = normalized[:i] + 'S' + normalized[i + 1:]
                break
    return normalized


class ParkingMonitor:
    def __init__(self, config):
        self.config = config
        self.ocr = PaddleOCR(**self.config['ocr_params'])
        self.model = YOLO(self.config['model_path'])
        self.polygon_points = []
        self.vehicle_records = {}
        self.detected_plates = set()
        self.plate_to_dwell_time = {}

        # Modified output directory structure
        self.output_dir = os.path.join('output folder', 'Parking_violation', datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, 'violations.txt')

    def point_in_polygon(self, x, y, polygon):
        n = len(polygon)
        inside = False
        px, py = polygon[-1]
        for i in range(n):
            x_i, y_i = polygon[i]
            if ((y_i > y) != (py > y)) and (x < (x_i - px) * (y - py) / (y_i - py) + px):
                inside = not inside
            px, py = x_i, y_i
        return inside

    def draw_polygon(self, frame):
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()

        cv2.namedWindow("Define Parking Area")
        cv2.setMouseCallback("Define Parking Area", mouse_callback)

        while True:
            temp_frame = frame.copy()
            if len(points) > 1:
                cv2.polylines(temp_frame, [np.array(points)], False, (0, 255, 0), 5)

            for p in points:
                cv2.circle(temp_frame, p, 10, (0, 0, 255), -1)

            cv2.imshow("Define Parking Area", temp_frame)
            cv2.waitKey(1)

            if len(points) == 4:
                break

        cv2.destroyWindow("Define Parking Area")
        return points

    def preprocess_roi(self, roi):
        """Preprocess the ROI to improve OCR accuracy"""
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        return sharpened

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Failed to read video")

        self.polygon_points = self.draw_polygon(frame)

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.config['frame_skip'] != 0:
                continue

            results = self.model.track(frame, persist=True, classes=[2, 3, 5, 7],
                                       conf=self.config['detection_confidence'])
            annotated = frame.copy()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().tolist()
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for track_id, box in zip(track_ids, boxes):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if self.point_in_polygon(cx, cy, self.polygon_points):
                        record = self.vehicle_records.setdefault(track_id, {
                            'entry': frame_count / fps,
                            'last_seen': frame_count / fps,
                            'violated': False
                        })

                        record['last_seen'] = frame_count / fps
                        dwell_time = record['last_seen'] - record['entry']

                        box_color = (0, 0, 255) if record['violated'] else (0, 255, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                        timer_text = f"{dwell_time:.1f}s"
                        cv2.putText(annotated, timer_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2)

                        if dwell_time > self.config['violation_duration'] and not record['violated']:
                            self.record_violation(track_id, frame, box, dwell_time)
                            cv2.putText(annotated, "Violation Detected", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.polylines(annotated, [np.array(self.polygon_points)], True, (0, 0, 255), 5)
            cv2.imshow("Monitoring", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.detected_plates:
            show_phone_input_gui(self.detected_plates, self.plate_to_dwell_time, violation_type="Illegal Parking")

    def record_violation(self, track_id, frame, box, dwell_time):
        self.vehicle_records[track_id]['violated'] = True
        x1, y1, x2, y2 = box

        height = y2 - y1
        roi_y1 = max(0, y2 - int(height / 2))
        roi_y2 = y2
        roi_x1 = max(0, x1)
        roi_x2 = x2
        plate_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        processed_roi = self.preprocess_roi(plate_roi)
        if processed_roi is None or processed_roi.shape[0] < 10 or processed_roi.shape[1] < 10:
            plate = "INVALID_ROI"
        else:
            try:
                ocr_result = self.ocr.ocr(processed_roi)
                if ocr_result and ocr_result[0]:
                    plate = " ".join(res[1][0] for res in ocr_result[0])
                    plate = normalize_plate_text(plate)
                else:
                    plate = "NO_TEXT_DETECTED"
            except Exception as e:
                plate = f"OCR_ERROR: {str(e)}"

        if plate not in ["INVALID_ROI", "NO_TEXT_DETECTED"] and not plate.startswith("OCR_ERROR"):
            self.detected_plates.add(plate)
            self.plate_to_dwell_time[plate] = dwell_time

            # Write to text file in the new directory
            with open(self.log_file, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp} - Vehicle ID: {track_id}, Plate: {plate}, Dwell Time: {dwell_time:.1f}s\n")

        print(f"Vehicle ID: {track_id}, Plate: {plate}, Dwell Time: {dwell_time:.1f}s")


def detect_illegal_parking(video_path):
    monitor = ParkingMonitor(CONFIG)
    monitor.process_video(video_path)