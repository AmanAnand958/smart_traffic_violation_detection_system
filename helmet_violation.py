# helmet_violation.py
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
from datetime import datetime
import re
import tkinter as tk
from tkinter import ttk
from utils import show_phone_input_gui  # Add this import

# Initialize PaddleOCR
ocr = PaddleOCR()

# Load YOLO Model
model = YOLO("models/best.pt")  # Change to your model path
names = model.names
print("Available classes:", names)  # Print available classes to verify

# Global variables for monitoring area selection
polygon_points = []

def draw_polygon(event, x, y, flags, param):
    global polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"Point added: {(x, y)}")

def get_monitoring_area(frame):
    global polygon_points
    clone = frame.copy()
    window_name = "Draw Monitoring Area (Press 'q' to finish)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon)

    while True:
        temp_frame = clone.copy()
        if polygon_points:
            for i, point in enumerate(polygon_points):
                cv2.circle(temp_frame, point, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(temp_frame, polygon_points[i - 1], point, (255, 0, 0), 2)
            if len(polygon_points) > 1:
                cv2.line(temp_frame, polygon_points[-1], polygon_points[0], (255, 0, 0), 2)

        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    return polygon_points

def perform_ocr(image_array):
    if image_array is None or image_array.size == 0:
        print("[OCR] ❌ Empty or invalid image.")
        return "UNKNOWN"

    results = ocr.ocr(image_array, rec=True)
    detected_text = []

    if results and results[0] is not None:
        for result in results[0]:
            text = result[1][0]
            detected_text.append(text)

    raw_text = ''.join(detected_text)
    cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    if cleaned_text:
        print(f"[OCR] Cleaned Number Plate: {cleaned_text}")
        return cleaned_text
    else:
        print(f"[OCR] ❌ No valid text found after cleaning. Raw Text: '{raw_text}'")
        return "UNKNOWN"

def save_to_txt(detected_plates, output_folder="output/helmet_violations"):
    os.makedirs(output_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y-%m-%d')
    txt_filename = os.path.join(output_folder, f"number_plates_{current_date}.txt")
    with open(txt_filename, 'w') as f:
        for plate in detected_plates:
            f.write(f"{plate}\n")
    print(f"✅ Number plates saved to: {txt_filename}")

def detect_helmet_violation(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error opening video file")
        return

    detected_plates = set()  # Use a set to store unique plates
    processed_track_ids = set()
    plate_to_violation = {}  # Map plates to violation data (True for no helmet)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to match model input size
        frame = cv2.resize(frame, (1020, 500))

        # YOLOv8 detection and tracking
        results = model.track(frame, persist=True)

        no_helmet_boxes = []
        numberplate_boxes = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                x1, y1, x2, y2 = box
                class_name = names[class_id]

                # Only process and display no-helmet and numberplate
                if class_name == 'no-helmet':
                    color = (0, 0, 255)  # Red for no-helmet
                    label = "No Helmet"
                    no_helmet_boxes.append((box, track_id))
                    # Draw bounding box and label for no-helmet only
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif class_name == 'numberplate':
                    color = (255, 0, 0)  # Blue for numberplate
                    label = "Number Plate"
                    numberplate_boxes.append((box, track_id))
                    # Draw bounding box and label for numberplate
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Skip helmet class completely - no drawing or processing

        # Process number plates when no-helmet detected
        if no_helmet_boxes:
            for numberplate_box, numberplate_track_id in numberplate_boxes:
                x1_np, y1_np, x2_np, y2_np = numberplate_box
                np_center = ((x1_np + x2_np) // 2, (y1_np + y2_np) // 2)

                for no_helmet_box, no_helmet_track_id in no_helmet_boxes:
                    x1_nh, y1_nh, x2_nh, y2_nh = no_helmet_box
                    nh_center = ((x1_nh + x2_nh) // 2, (y1_nh + y2_nh) // 2)

                    # Calculate distance between number plate and no-helmet
                    distance = np.linalg.norm(np.array(np_center) - np.array(nh_center))
                    if distance < 200 and numberplate_track_id not in processed_track_ids:
                        crop = frame[y1_np:y2_np, x1_np:x2_np]
                        if crop.size == 0:
                            continue

                        crop = cv2.resize(crop, (120, 85))
                        text = perform_ocr(crop)

                        if text != "UNKNOWN":
                            detected_plates.add(text)
                            processed_track_ids.add(numberplate_track_id)
                            plate_to_violation[text] = True  # No helmet detected

        # Display frame with bounding boxes and labels
        cv2.imshow("Traffic Violation Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Save detected plates to text file
    save_to_txt(detected_plates)

    # Show phone input GUI and send messages
    if detected_plates:
        show_phone_input_gui(detected_plates, plate_to_violation, violation_type="Helmet Violation")