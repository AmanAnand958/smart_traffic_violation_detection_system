# signal_violation.py
import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
import os
from datetime import datetime
from paddleocr import PaddleOCR
import logging
from utils import show_phone_input_gui  # Add this import
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize PaddleOCR for number plate recognition
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Global list to store polygon points (drawn on the original frame)
polygon_points = []


def draw_polygon(event, x, y, flags, param):
    """Mouse callback function to record polygon vertices."""
    global polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"Point added: {(x, y)}")


def get_monitoring_area(frame):
    """
    Displays the given frame and lets the user draw a polygon by clicking on the image.
    Press any key when finished.
    """
    global polygon_points
    polygon_points = []
    clone = frame.copy()
    window_name = "Draw Monitoring Area (Press any key to finish)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon)

    while True:
        temp_frame = clone.copy()
        # Draw the current points and connecting lines
        if polygon_points:
            for i, point in enumerate(polygon_points):
                cv2.circle(temp_frame, point, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(temp_frame, polygon_points[i - 1], point, (255, 0, 0), 2)
            # Optionally, draw a closing line of the polygon
            if len(polygon_points) > 1:
                cv2.line(temp_frame, polygon_points[-1], polygon_points[0], (255, 0, 0), 2)

        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Break on any key press (255 means no key pressed)
            break

    cv2.destroyWindow(window_name)
    return polygon_points


# Integrated process_frame function from test1.py
def process_frame(frame):
    lower_range = np.array([58, 97, 222])  # Green color range
    upper_range = np.array([179, 255, 255])
    lower_range1 = np.array([0, 43, 184])  # Red color range
    upper_range1 = np.array([56, 132, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for both color ranges
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask1 = cv2.inRange(hsv, lower_range1, upper_range1)

    # Combine the two masks
    combined_mask = cv2.bitwise_or(mask, mask1)

    # Threshold the combined mask
    _, final_mask = cv2.threshold(combined_mask, 254, 255, cv2.THRESH_BINARY)
    detected_label = None

    # Find contours
    cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)

            # Calculate the center point of the rectangle
            cx = x + w // 2
            cy = y + h // 2
            if cx < 915:
                # Determine the color of the contour
                if cv2.countNonZero(mask[y:y + h, x:x + w]) > 0:  # Green color range
                    color = (0, 255, 0)  # Green color for the rectangle
                    text_color = (0, 255, 0)  # Green text
                    label = "GREEN"
                elif cv2.countNonZero(mask1[y:y + h, x:x + w]) > 0:  # Red color range
                    color = (0, 0, 255)  # Red color for the rectangle
                    text_color = (0, 0, 255)  # Red text
                    label = "RED"
                else:
                    continue

                detected_label = label
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw the center point
                cv2.circle(frame, (cx, cy), 1, (255, 0, 0), -1)  # Draw a small blue circle at the center

                # Display text
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    return frame, detected_label


# Integrated Tracker class from tracker.py
class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


def detect_signal_violation(video_path):
    # Initialize YOLO model
    model = YOLO("models/yolov8s.pt")

    # Define valid class IDs for cars and bikes (based on COCO dataset)
    valid_class_ids = {2, 3}  # 1 = bicycle, 2 = car
    # If "bikes" meant motorcycles, use {2, 3} where 3 = motorcycle

    # Initialize tracker
    tracker = Tracker()

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video source {video_path}")
        return

    # Grab the first frame to set up the monitoring area (original size)
    ret, first_frame = cap.read()
    if not ret:
        logging.error("Error: Could not read frame from video")
        return

    original_h, original_w, _ = first_frame.shape
    logging.info(f"Original frame size: {original_w}x{original_h}")

    # Let the user draw the monitoring area on the first frame
    logging.info("Please draw the monitoring area on the displayed frame.")
    monitoring_area = get_monitoring_area(first_frame)
    logging.info(f"Monitoring area selected (original coordinates): {monitoring_area}")

    # Define target size used in processing
    target_width, target_height = 1020, 600
    scale_x = target_width / original_w
    scale_y = target_height / original_h
    scaled_monitoring_area = [(int(x * scale_x), int(y * scale_y)) for (x, y) in monitoring_area]
    logging.info(f"Scaled monitoring area for processing: {scaled_monitoring_area}")

    # Create output directory
    today_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join('img', today_date)
    os.makedirs(output_dir, exist_ok=True)

    detected_plates = set()  # Store unique plates
    plate_to_violation = {}  # Map plates to violation data (True for signal violation)
    count = 0
    tracked_ids = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 2 != 0:
            continue  # Process every second frame for performance optimization

        # Resize frame to target size
        frame = cv2.resize(frame, (target_width, target_height))
        processed_frame, detected_label = process_frame(frame)

        # YOLO object detection
        results = model(frame)
        detections = pd.DataFrame(results[0].boxes.data).astype("float")

        if not detections.empty and len(detections.columns) > 5:
            # Filter detections to only include cars (class ID 2) and bikes (class ID 1)
            boxes = [
                [int(x1), int(y1), int(x2), int(y2)]
                for x1, y1, x2, y2, _, cls_id in detections.values
                if int(cls_id) in valid_class_ids
            ]
        else:
            boxes = []

        # Update tracker with new detections
        tracked_objects = tracker.update(boxes)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Check if the center point is inside the scaled monitoring area
            in_area = cv2.pointPolygonTest(np.array(scaled_monitoring_area, np.int32), (cx, cy), False) >= 0

            if in_area:
                # Find the class ID for this object from detections
                for _, row in detections.iterrows():
                    if (int(row[0]) == x1 and int(row[1]) == y1 and
                            int(row[2]) == x2 and int(row[3]) == y2):
                        class_id = int(row[5])
                        break
                else:
                    class_id = None

                # Check if it's a car or bike and if the signal is red
                if class_id in valid_class_ids and detected_label == "RED":
                    if obj_id not in tracked_ids:
                        tracked_ids.append(obj_id)
                        timestamp = datetime.now().strftime('%H-%M-%S')
                        image_path = os.path.join(output_dir, f"{timestamp}_{obj_id}.jpg")
                        cv2.imwrite(image_path, frame)
                        number_plate_text = recognize_number_plate(frame, x1, y1, x2, y2)
                        if number_plate_text:
                            detected_plates.add(number_plate_text)
                            plate_to_violation[number_plate_text] = True  # Signal violation detected

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'VIOLATION {obj_id}', (x1, y1), 1, 1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{obj_id}', (x1, y1), 1, 1)

        # Draw the scaled monitoring area on the frame
        cv2.polylines(frame, [np.array(scaled_monitoring_area, np.int32)], True, (0, 255, 0), 2)
        cv2.imshow("Traffic Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Show phone input GUI and send messages
    if detected_plates:
        show_phone_input_gui(detected_plates, plate_to_violation, violation_type="Signal Violation")


def recognize_number_plate(frame, x1, y1, x2, y2):
    """Extract number plate text using PaddleOCR."""
    h, w, _ = frame.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    number_plate_roi = frame[y1:y2, x1:x2].copy()
    result = ocr.ocr(number_plate_roi, cls=True)
    extracted_text = ""
    if result and result[0]:
        extracted_text = " ".join([word_info[1][0] for line in result for word_info in line])
    return extracted_text.strip()