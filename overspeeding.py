# overspeeding.py
import cv2
import numpy as np
import os
import tkinter as tk
import tkinter.simpledialog as simpledialog
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from paddleocr import PaddleOCR
from utils import show_phone_input_gui

# Load YOLO model (nano for performance)
vehicle_model = YOLO("models/yolov8n.pt").to('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')

# Initialize DeepSort tracker
tracker = DeepSort(max_age=50, n_init=3, nms_max_overlap=0.5)

# Initialize PaddleOCR with GPU support
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)

# Constants
SPEED_LIMIT = 60
OUTPUT_DIR = "output/speed_violation"
TARGET_WIDTH = 960
FRAME_SKIP = 3  # Process every 3rd frame
DETECTION_INTERVAL = 5

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_plate_image(image):
    """Preprocess image for OCR with contrast enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    _, thresh = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def detect_and_recognize_plate(frame, vehicle_bbox, ocr_results):
    """Recognize number plate with validation and store results."""
    x1, y1, x2, y2 = map(int, vehicle_bbox)
    if x2 - x1 < 50 or y2 - y1 < 20:
        return "INVALID_BBOX"
    vehicle_img = frame[y1:y2, x1:x2]
    if vehicle_img.size == 0:
        return "EMPTY_IMAGE"
    preprocessed = preprocess_plate_image(vehicle_img)
    ocr_result = ocr.ocr(preprocessed, cls=True)
    if ocr_result and ocr_result[0]:
        text = ''.join([word[1][0] for word in ocr_result[0]]).strip()
        confidence = max(word[1][1] for word in ocr_result[0]) if ocr_result[0] else 0
        if text and confidence > 0.6:
            ocr_results.append((text, confidence))
        return text if text else "UNKNOWN"
    return "UNKNOWN"

def select_best_plate(ocr_results):
    """Select the highest-confidence plate across frames."""
    return max(ocr_results, key=lambda x: x[1])[0] if ocr_results else None

def calculate_real_distance(frame, points):
    """Map pixel distances to real-world using perspective transform."""
    src_pts = np.float32(points)
    dst_pts = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]])  # Meters
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def define_measurement_lines(frame):
    points = []
    window_name = "Click 4 points for two lines (top-left, top-right, bottom-left, bottom-right)"
    cv2.namedWindow(window_name)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
            elif len(points) == 4:
                cv2.line(frame, points[2], points[3], (0, 255, 0), 2)
            cv2.imshow(window_name, frame)

    cv2.setMouseCallback(window_name, mouse_callback)
    while len(points) < 4:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)
    return points if len(points) == 4 else (None, None, None, None)

def detect_speed_violation(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not read video.")
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    aspect_ratio = frame.shape[1] / frame.shape[0]
    target_height = int(TARGET_WIDTH / aspect_ratio)
    frame = cv2.resize(frame, (TARGET_WIDTH, target_height))

    line1_start, line1_end, line2_start, line2_end = define_measurement_lines(frame.copy())
    if line1_start is None:
        print("Error: Measurement lines not defined.")
        cap.release()
        return

    root = tk.Tk()
    root.withdraw()
    real_distance = simpledialog.askfloat("Input", "Enter real-world distance between lines (meters):")
    root.destroy()
    if not real_distance or real_distance <= 0:
        print("Error: Invalid distance.")
        cap.release()
        return

    matrix = calculate_real_distance(frame, [line1_start, line1_end, line2_start, line2_end])
    tracking_data = {}
    violation_counts = {}  # For temporal consistency
    violation_logged = {}
    ocr_results_per_id = {}  # Multi-frame OCR
    fps = cap.get(cv2.CAP_PROP_FPS)
    detected_plates = set()
    plate_to_speed = {}
    frame_count = 0

    line1_y = (line1_start[1] + line1_end[1]) // 2
    line2_y = (line2_start[1] + line2_end[1]) // 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:  # Skip frames
            continue

        frame = cv2.resize(frame, (TARGET_WIDTH, target_height))
        cv2.line(frame, line1_start, line1_end, (0, 255, 0), 2)
        cv2.line(frame, line2_start, line2_end, (0, 255, 0), 2)

        if frame_count % DETECTION_INTERVAL == 0:
            results = vehicle_model(frame)
            detections = [(box, score, None) for box, score in
                          zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy())]
            current_tracks = tracker.update_tracks(detections, frame=frame)

        for track in current_tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()
            center_x, center_y = get_center(bbox)

            if track_id not in tracking_data:
                tracking_data[track_id] = {'crossed_line1_frame': None, 'speed': None}
                ocr_results_per_id[track_id] = []

            data = tracking_data[track_id]
            if data['crossed_line1_frame'] is None and center_y > line1_y:
                data['crossed_line1_frame'] = frame_count

            if data['crossed_line1_frame'] is not None and center_y > line2_y:
                frames_between = frame_count - data['crossed_line1_frame']
                time_seconds = frames_between / fps
                if time_seconds > 0:
                    # Transform centers to real-world coords
                    pt1 = np.array([[center_x, line1_y]], dtype=np.float32)
                    pt2 = np.array([[center_x, line2_y]], dtype=np.float32)
                    pt1_transformed = cv2.perspectiveTransform(pt1[None, :, :], matrix)[0][0]
                    pt2_transformed = cv2.perspectiveTransform(pt2[None, :, :], matrix)[0][0]
                    pixel_distance = np.linalg.norm(pt2_transformed - pt1_transformed)
                    speed_mps = real_distance * pixel_distance / time_seconds
                    speed_kmh = speed_mps * 3.6
                    data['speed'] = speed_kmh

                    if speed_kmh > SPEED_LIMIT:
                        violation_counts[track_id] = violation_counts.get(track_id, 0) + 1
                        if violation_counts[track_id] >= 3 and track_id not in violation_logged:  # Temporal consistency
                            plate = detect_and_recognize_plate(frame, bbox, ocr_results_per_id[track_id])
                            if plate != "UNKNOWN" and plate not in ["INVALID_BBOX", "EMPTY_IMAGE"]:
                                final_plate = select_best_plate(ocr_results_per_id[track_id])
                                if final_plate:
                                    detected_plates.add(final_plate)
                                    speed_over_limit = int(speed_kmh - SPEED_LIMIT)
                                    plate_to_speed[final_plate] = speed_over_limit
                                    violation_logged[track_id] = True

            if data['speed'] is not None:
                color = (0, 0, 255) if data['speed'] > SPEED_LIMIT else (0, 255, 0)
                label = f"ID {track_id}: {int(data['speed'])} km/h"
                cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

        cv2.imshow("Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if detected_plates:
        show_phone_input_gui(detected_plates, plate_to_speed, violation_type="Overspeed")