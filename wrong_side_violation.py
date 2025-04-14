import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
import re
from utils import show_phone_input_gui

# Global variables for direction selection
direction_points = []
frame_copy = None


def get_user_direction(frame):
    """
    Allows the user to define the correct direction by clicking two points on the frame.
    Returns the direction vector as (dx, dy).
    """
    global direction_points, frame_copy
    direction_points = []
    frame_copy = frame.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(direction_points) < 2:
            direction_points.append((x, y))
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            if len(direction_points) == 2:
                cv2.arrowedLine(frame_copy, direction_points[0], direction_points[1], (0, 255, 0), 2)
            cv2.imshow("Select Direction", frame_copy)

    cv2.imshow("Select Direction", frame_copy)
    cv2.setMouseCallback("Select Direction", click_event)
    cv2.waitKey(0)
    cv2.destroyWindow("Select Direction")

    if len(direction_points) != 2:
        raise ValueError("You need to click exactly two points to define the direction.")
    start, end = direction_points
    return (end[0] - start[0], end[1] - start[1])


def normalize_plate(text):
    """
    Normalizes the license plate text by removing spaces and converting to uppercase.
    """
    return re.sub(r'\s+', '', text).upper()


def preprocess_image(image):
    """
    Preprocesses the image to enhance text readability for OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)  # Slight contrast enhancement
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduce noise
    # Removed adaptive thresholding to preserve text details
    return gray


def select_best_plate(ocr_results):
    """
    Selects the best plate from multiple OCR results based on length, confidence, and alphanumeric content.
    """
    if not ocr_results:
        return None

    def score_plate(text, confidence, displacement):
        length_score = len(text)
        alpha_count = sum(c.isalpha() for c in text)
        digit_count = sum(c.isdigit() for c in text)
        pattern_score = min(alpha_count, digit_count) * 2
        return length_score + pattern_score + confidence * 10 - displacement * 0.1

    best_result = max(ocr_results, key=lambda x: score_plate(x[0], x[1], x[2]) if x[1] > 0.7 else 0,
                      default=(None, 0, 0))
    return best_result[0] if best_result[0] and best_result[1] > 0.7 else None


def wrong_side_driving(video_path,
                       vehicle_model_path="models/yolov8n.pt",
                       min_frames=15,
                       max_history_length=15,
                       frame_skip=2,
                       stationary_threshold=0.1,
                       ocr_confidence=0.7,
                       ocr_interval=3,
                       slow_displacement_threshold=5):
    """
    Detects vehicles moving against the user-defined direction and stores unique license plates.
    Performs OCR multiple times per vehicle ID with improved detection.
    """
    # Validate input files
    if not os.path.exists(video_path) or not os.path.exists(vehicle_model_path):
        raise FileNotFoundError(f"File not found: {video_path} or {vehicle_model_path}")

    # Load models
    vehicle_model = YOLO(vehicle_model_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, det_db_box_thresh=0.5, det_db_unclip_ratio=2.0)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get direction from first frame
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame.")
    correct_direction_vector = get_user_direction(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Tracking variables
    track_history = {}
    detected_plates = set()
    plate_to_violation = {}
    normalized_plates_set = set()
    ocr_results_per_id = {}
    frame_count = 0
    vehicle_classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        original_frame = frame.copy()
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        display_frame = frame.copy()

        # Detect and track vehicles
        results = vehicle_model.track(frame, persist=True, conf=0.5, iou=0.5)
        if results and len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if box.id is None or box.cls is None:
                        continue

                    class_id = int(box.cls.item())
                    if class_id not in vehicle_classes:
                        continue

                    track_id = int(box.id.item())
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    x_start = max(0, int(x - w / 2))
                    y_start = max(0, int(y - h / 2))
                    x_end = min(frame.shape[1], int(x + w / 2))
                    y_end = min(frame.shape[0], int(y + h / 2))
                    center_x = x_start + (x_end - x_start) / 2
                    center_y = y_start + (y_end - y_start) / 2

                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append((center_x, center_y))
                    if len(track_history[track_id]) > max_history_length:
                        track_history[track_id].pop(0)

                    # Analyze movement
                    if len(track_history[track_id]) >= min_frames:
                        first_pos = track_history[track_id][0]
                        last_pos = track_history[track_id][-1]
                        displacement = np.linalg.norm(np.array(last_pos) - np.array(first_pos))
                        threshold = stationary_threshold * w

                        print(f"Track ID: {track_id}, Displacement: {displacement:.2f}, Threshold: {threshold:.2f}")

                        if displacement < threshold:
                            color = (255, 0, 0)  # Blue
                            label = f"ID: {track_id} (Stationary)"
                        else:
                            displacement_vector = (last_pos[0] - first_pos[0], last_pos[1] - first_pos[1])
                            dot_product = (displacement_vector[0] * correct_direction_vector[0] +
                                           displacement_vector[1] * correct_direction_vector[1])
                            if dot_product < -0.1:
                                color = (0, 0, 255)  # Red
                                label = f"ID: {track_id} (Wrong)"

                                # Calculate recent displacement for frame selection
                                recent_displacement = float('inf')
                                if len(track_history[track_id]) >= 2:
                                    prev_pos = track_history[track_id][-2]
                                    current_pos = track_history[track_id][-1]
                                    recent_displacement = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))

                                # Perform OCR if displacement is small (clearer frame) and interval allows
                                if frame_count % ocr_interval == 0 and recent_displacement < slow_displacement_threshold:
                                    scale_factor = 2
                                    orig_x_start = max(0, int(x_start * scale_factor - w * 0.75))  # Increased padding
                                    orig_y_start = max(0, int(y_start * scale_factor - h * 0.75))
                                    orig_x_end = min(original_frame.shape[1], int(x_end * scale_factor + w * 0.75))
                                    orig_y_end = min(original_frame.shape[0], int(y_end * scale_factor + h * 0.75))

                                    vehicle_image = original_frame[orig_y_start:orig_y_end, orig_x_start:orig_x_end]
                                    if vehicle_image.size == 0:
                                        continue

                                    vehicle_image = preprocess_image(vehicle_image)
                                    cv2.imwrite(f"crop_{track_id}_{frame_count}.jpg", vehicle_image)

                                    ocr_result = ocr.ocr(vehicle_image, cls=True, det=True)
                                    print(f"Raw OCR Result for Track ID {track_id}, Frame {frame_count}: {ocr_result}")
                                    if ocr_result and ocr_result[0]:
                                        # Combine all detected text into a single string
                                        full_text = "".join(
                                            line[1][0] for line in ocr_result[0] if line[1][1] > ocr_confidence)
                                        confidence = max(
                                            line[1][1] for line in ocr_result[0] if line[1][1] > ocr_confidence)
                                        if full_text:
                                            if track_id not in ocr_results_per_id:
                                                ocr_results_per_id[track_id] = []
                                            ocr_results_per_id[track_id].append(
                                                (full_text, confidence, recent_displacement))

                                # Update label with the best plate
                                if track_id in ocr_results_per_id:
                                    best_plate = select_best_plate(ocr_results_per_id[track_id])
                                    if best_plate and best_plate not in normalized_plates_set:
                                        normalized_plates_set.add(normalize_plate(best_plate))
                                        detected_plates.add(best_plate)
                                        plate_to_violation[track_id] = best_plate
                                        label += f" Plate: {best_plate}"
                            else:
                                color = (0, 255, 0)  # Green
                                label = f"ID: {track_id} (Right)"

                        # Draw bounding box and label
                        cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), color, 1)
                        cv2.putText(display_frame, label, (x_start, y_start - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if frame_count % (frame_skip * 2) == 0:
            cv2.imshow("Wrong Side Driving Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    final_plates = {plate for track_id, plate in plate_to_violation.items()}
    if final_plates:
        show_phone_input_gui(final_plates, {p: True for p in final_plates}, violation_type="Wrong Side Driving")
    else:
        print("No wrong-side driving vehicles with valid plates detected.")


if __name__ == "__main__":
    wrong_side_driving("path/to/video.mp4")