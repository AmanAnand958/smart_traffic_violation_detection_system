# main.py (Overloading Module)
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
import numpy as np
from utils import show_phone_input_gui  # Import the shared utility function

def intersection_area(box1, box2):
    """Calculate the intersection area of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height

def is_rider(person_box, bike_box):
    """Check if a person is a rider on a bike based on center and intersection area."""
    center_x = (person_box[0] + person_box[2]) // 2
    center_y = (person_box[1] + person_box[3]) // 2
    if bike_box[0] <= center_x <= bike_box[2] and bike_box[1] <= center_y <= bike_box[3]:
        intersection = intersection_area(person_box, bike_box)
        person_area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
        ratio = intersection / person_area
        return ratio > 0.5
    return False

def count_riders(bike_box, people_boxes):
    """Count riders on a bike, ensuring each person is counted once."""
    riders = 0
    counted_people = set()
    for person in people_boxes:
        person_tuple = tuple(person.tolist())
        if person_tuple not in counted_people and is_rider(person, bike_box):
            riders += 1
            counted_people.add(person_tuple)
    return riders

def scale_up(resized_box, original_width, original_height):
    """Scale up bounding box coordinates from resized frame to original frame."""
    x1, y1, x2, y2 = resized_box
    scale_x = original_width / 640
    scale_y = original_height / 480
    return [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]

def extract_plate_region(bike_roi):
    """Extract potential license plate region from bike ROI using contour analysis."""
    gray = cv2.cvtColor(bike_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 1.5 < aspect_ratio < 4.0 and w > 50 and h > 20:
            plate_roi = bike_roi[y:y + h, x:x + w]
            return plate_roi
    return bike_roi

def preprocess_plate_image(plate_roi):
    """Preprocess license plate image for OCR with enhanced resizing and contrast."""
    height, width = plate_roi.shape[:2]
    if height < 300:
        scale_factor = 300 / height
        plate_roi = cv2.resize(plate_roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    plate_roi = cv2.bilateralFilter(plate_roi, 9, 50, 50)
    gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    gray_plate = cv2.convertScaleAbs(gray_plate, alpha=2.0, beta=0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_clahe = clahe.apply(gray_plate)
    thresh_plate = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 7, 2)
    return thresh_plate

def recognize_plate_text(thresh_plate, ocr):
    """Recognize text from preprocessed license plate image using PaddleOCR."""
    result = ocr.ocr(thresh_plate, cls=True, det=True)
    if result and result[0]:
        all_text = "".join([item[1][0] for item in result[0]])
        cleaned_text = ''.join(c for c in all_text if c.isalnum())
        return cleaned_text if len(cleaned_text) >= 6 else None
    return None

def extract_plate_from_regions(regions, ocr):
    """Attempt to extract a valid license plate from stored regions."""
    for frame_count, bike_roi in regions:
        plate_roi = extract_plate_region(bike_roi)
        if plate_roi.size > 0:
            thresh_plate = preprocess_plate_image(plate_roi)
            plate_text = recognize_plate_text(thresh_plate, ocr)
            if plate_text:
                return plate_text
    return None

def detect_overloading(video_path):
    """Detect overloaded bikes, extract plates, and trigger phone input GUI."""
    output_dir = "output/overloading"
    debug_dir = "output/debug"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    model = YOLO('models/yolov8m.pt')
    ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_batch_num=1)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detected_plates = set()  # Store unique license plates
    plate_to_riders = {}     # Map plates to number of riders
    track_id_to_regions = {} # Store bike ROIs for each track ID
    track_id_to_plate = {}   # Map track IDs to plates
    track_id_to_riders = {}  # Map track IDs to rider counts
    violating_ids = set()    # Track IDs of violating bikes

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_resized = cv2.resize(frame, (640, 480))
        results = model.track(frame_resized, persist=True)
        annotated_frame = results[0].plot()

        bikes = [(box, int(id)) for box, id in zip(results[0].boxes.xyxy.cpu().numpy(),
                                                   results[0].boxes.id.cpu().numpy())
                 if results[0].boxes.cls.cpu().numpy()[list(results[0].boxes.id.cpu().numpy()).index(id)] == 3]
        people = [box for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(),
                                          results[0].boxes.cls.cpu().numpy()) if cls == 0]

        for bike_box_resized, track_id in bikes:
            riders = count_riders(bike_box_resized, people)
            bike_box_original = scale_up(bike_box_resized, frame_width, frame_height)
            bike_roi = frame[max(0, bike_box_original[1]):min(frame_height, bike_box_original[3]),
                             max(0, bike_box_original[0]):min(frame_width, bike_box_original[2])]

            if track_id not in track_id_to_regions:
                track_id_to_regions[track_id] = []
            track_id_to_regions[track_id].append((frame_count, bike_roi))

            if riders >= 3:  # Overloading threshold (3 or more riders)
                violating_ids.add(track_id)
                track_id_to_riders[track_id] = riders

                if track_id not in track_id_to_plate:
                    plate_roi = extract_plate_region(bike_roi)
                    if plate_roi.size > 0:
                        cv2.imwrite(os.path.join(debug_dir, f"plate_roi_{track_id}_{frame_count}.jpg"), plate_roi)
                        thresh_plate = preprocess_plate_image(plate_roi)
                        cv2.imwrite(os.path.join(debug_dir, f"thresh_plate_{track_id}_{frame_count}.jpg"), thresh_plate)

                        plate_text = recognize_plate_text(thresh_plate, ocr)
                        if plate_text:
                            track_id_to_plate[track_id] = plate_text
                            detected_plates.add(plate_text)
                            plate_to_riders[plate_text] = riders
                            print(f"Frame {frame_count}, Track ID {track_id}: Detected plate - {plate_text}, Riders: {riders}")

        cv2.imshow('Overloading Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Fallback: Try to extract plates for violating bikes that weren't identified during processing
    for track_id in violating_ids:
        if track_id not in track_id_to_plate:
            plate_text = extract_plate_from_regions(track_id_to_regions[track_id], ocr)
            if plate_text:
                track_id_to_plate[track_id] = plate_text
                detected_plates.add(plate_text)
                plate_to_riders[plate_text] = track_id_to_riders[track_id]
                print(f"Track ID {track_id}: Fallback detected plate - {plate_text}, Riders: {track_id_to_riders[track_id]}")

    # Save detected plates to a text file
    plates_file = os.path.join(output_dir, "detected_plates.txt")
    with open(plates_file, 'w') as f:
        for plate in detected_plates:
            if len(plate) >= 6:
                f.write(f"{plate}\n")

    cap.release()
    cv2.destroyAllWindows()

    # Show phone input GUI and send messages
    if detected_plates:
        show_phone_input_gui(detected_plates, plate_to_riders, violation_type="Overloading")

if __name__ == "__main__":
    video_path = "path_to_your_video.mp4"  # Replace with your video file path
    detect_overloading(video_path)