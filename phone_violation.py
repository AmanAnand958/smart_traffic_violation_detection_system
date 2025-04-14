import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import math
import os
from paddleocr import PaddleOCR
from utils import show_phone_input_gui

# Initialize YOLO
model = YOLO('models/yolov8m.pt')

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_palm_center(hand_landmarks, img_width, img_height):
    """Calculate the palm center from hand landmarks."""
    x_sum = y_sum = 0
    for i in range(5):  # Landmarks 0-4 (wrist to thumb base)
        x_sum += hand_landmarks.landmark[i].x * img_width
        y_sum += hand_landmarks.landmark[i].y * img_height
    return int(x_sum / 5), int(y_sum / 5)

def preprocess_roi(roi):
    """Preprocess ROI for better OCR accuracy."""
    if roi.size == 0:
        return None
    roi_resized = cv2.resize(roi, (300, 100), interpolation=cv2.INTER_AREA)
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    roi_enhanced = cv2.equalizeHist(roi_gray)
    return roi_enhanced

def normalize_plate_text(plate_text):
    """Normalize license plate text by removing hyphens, converting to uppercase, and applying specific character replacements."""
    if not plate_text:
        return None
    # Remove hyphens and any whitespace, convert to uppercase
    normalized = plate_text.replace("-", "").replace(" ", "").upper()
    # Replace 'O' with 'D' in the first position
    if normalized and normalized[0] == 'O':
        normalized = 'D' + normalized[1:]
    # Replace the first 'S' with '5' (after the first character)
    for i in range(1, len(normalized)):
        if normalized[i] == 'S':
            normalized = normalized[:i] + '5' + normalized[i + 1:]
            break  # Only replace the first 'S'
    # Replace the second '5' with 'S'
    count = 0
    for i in range(len(normalized)):
        if normalized[i] == '5':
            count += 1
            if count == 2:  # When we find the second '5'
                normalized = normalized[:i] + 'S' + normalized[i + 1:]
                break
    return normalized

def extract_plate_text(roi):
    """Extract number plate text from ROI using PaddleOCR and normalize it."""
    preprocessed_roi = preprocess_roi(roi)
    if preprocessed_roi is None:
        print("Preprocessed ROI is empty, skipping OCR")
        return None
    result = ocr.ocr(preprocessed_roi, cls=True)
    if result and result[0]:
        plate_text = "".join([item[1][0] for item in result[0]])
        normalized_plate = normalize_plate_text(plate_text)
        if normalized_plate:
            print(f"OCR detected plate: {plate_text}, Normalized: {normalized_plate}")
            return normalized_plate
        print("Normalized plate text is empty")
        return None
    print("OCR failed to detect plate text")
    return None

def detect_mobile_phone_usage(video_path):
    """Detect if a hand is above the middle line and save violating plates to a text file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")

    output_dir = "output/mobile_phone_detection"
    os.makedirs(output_dir, exist_ok=True)
    violation_file = os.path.join(output_dir, "violations.txt")

    detected_plates = set()          # Store unique plates of violating vehicles
    vehicle_plates = {}              # Map vehicle IDs to their plates
    plate_to_violation = {}          # Map plates to violation data (True for phone usage)
    vehicle_violation_status = {}    # Map vehicle IDs to violation status (True if violated at least once)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print(f"Processing frame {frame_count}")

        frame_resized = cv2.resize(frame, (640, 480))
        results = model.track(frame_resized, persist=True)

        people = [(box.xyxy[0].tolist(), int(box.id)) for box in results[0].boxes if int(box.cls) == 0 and box.id is not None]
        two_wheelers = [(box.xyxy[0].tolist(), int(box.id)) for box in results[0].boxes if int(box.cls) in [1, 3] and box.id is not None]

        scale_x, scale_y = frame_width / 640, frame_height / 480
        people = [([int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)], id) for [x, y, w, h], id in people]
        two_wheelers = [([int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)], id) for [x, y, w, h], id in two_wheelers]

        # Try to extract plates for all detected vehicles in every frame
        for vehicle_box, vehicle_id in two_wheelers:
            vx, vy, vx2, vy2 = vehicle_box
            vh = vy2 - vy
            plate_roi = frame[max(0, vy + vh // 2):min(frame_height, vy2), max(0, vx):min(frame_width, vx2)]
            plate_text = extract_plate_text(plate_roi)
            if plate_text:
                vehicle_plates[vehicle_id] = plate_text
                print(f"Stored plate {plate_text} for vehicle ID {vehicle_id}")

        # Process each person
        for person_box, person_id in people:
            px, py, px2, py2 = person_box
            pw, ph = px2 - px, py2 - py

            # Associate person with vehicle using IoU
            max_iou = 0
            vehicle_id = None
            vehicle_box = None
            for v_box, v_id in two_wheelers:
                iou = calculate_iou(person_box, v_box)
                if iou > max_iou:
                    max_iou = iou
                    vehicle_id = v_id
                    vehicle_box = v_box
            if max_iou < 0.3:
                print(f"Frame {frame_count}: Person ID {person_id} not associated with any vehicle (IoU < 0.3)")
                continue

            # Expand ROI for hand detection by 50 pixels
            margin = 50
            person_roi = frame[max(0, py - margin):min(frame_height, py2 + margin),
                              max(0, px - margin):min(frame_width, px2 + margin)]
            if person_roi.size == 0:
                print(f"Frame {frame_count}: Person ROI is empty, skipping hand detection")
                continue

            try:
                rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                print(f"Frame {frame_count}: Error converting to RGB - {e}")
                continue

            with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
                hand_results = hands.process(rgb_roi)
                if not hand_results:
                    print(f"Frame {frame_count}: Hand detection failed")
                    continue

                hand_centers = []
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        palm_center = get_palm_center(hand_landmarks, person_roi.shape[1], person_roi.shape[0])
                        palm_center = (palm_center[0] + max(0, px - margin), palm_center[1] + max(0, py - margin))
                        hand_centers.append(palm_center)
                        mp_drawing.draw_landmarks(person_roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                middle_line_y = py + ph // 2
                violation = False
                for hand_x, hand_y in hand_centers:
                    if hand_y < middle_line_y:
                        violation = True
                        print(f"Violation detected in frame {frame_count} for vehicle ID {vehicle_id}: Hand above middle line")
                        vehicle_violation_status[vehicle_id] = True
                        break

                # Display based on violation status
                if vehicle_violation_status.get(vehicle_id, False):
                    cv2.rectangle(frame, (px, py), (px2, py2), (0, 0, 255), 2)  # Red for violation
                    plate_display = vehicle_plates.get(vehicle_id, "No Plate")
                    cv2.putText(frame, f"Violation - ID {vehicle_id}: {plate_display}", (px, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (px, py), (px2, py2), (0, 255, 0), 2)  # Green for no violation

                cv2.line(frame, (px, middle_line_y), (px2, middle_line_y), (0, 255, 0), 1)

        # Associate plates with violating vehicles and save to file
        for vehicle_id in vehicle_violation_status:
            if vehicle_violation_status[vehicle_id] and vehicle_id in vehicle_plates:
                plate_text = vehicle_plates[vehicle_id]
                if plate_text and plate_text.strip():
                    detected_plates.add(plate_text)
                    plate_to_violation[plate_text] = True
                    print(f"Associated plate {plate_text} with violating vehicle ID {vehicle_id} in frame {frame_count}")

        cv2.imshow('Mobile Phone Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save detected plates to text file
    if detected_plates:
        with open(violation_file, 'w') as f:
            for plate in detected_plates:
                f.write(f"{plate}\n")
        print(f"Violating license plates saved to {violation_file}")

    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

    # Show phone input GUI and send messages
    if detected_plates:
        show_phone_input_gui(detected_plates, plate_to_violation, violation_type="Phone Violation")

if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"  # Replace with your video path
    detect_mobile_phone_usage(video_path)