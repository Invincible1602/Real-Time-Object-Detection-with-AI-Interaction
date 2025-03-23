from ultralytics import YOLO
import cv2 as cv
import requests
import time
import pyttsx3
from concurrent.futures import ThreadPoolExecutor

# Initialize TTS engine
engine = pyttsx3.init()

# Load both models
model_custom = YOLO("best.pt")
model_coco = YOLO("yolov5n.pt")

# Set custom model class names and update internal names dictionary
hotel_names = ["ac", "bed", "c", "chair", "clock", "cup", "sofa", "tv", "tvmonitor"]
model_custom.model.names = {i: name for i, name in enumerate(hotel_names)}

# Load COCO names for model_coco
with open("coco.names", "r") as f:
    coco_names = [line.strip() for line in f.readlines()]
model_coco.model.names = {i: name for i, name in enumerate(coco_names)}

# FAQ Chatbot API endpoint
FAQ_API_URL = "http://localhost:8000/faq/"

def query_faq_chatbot(query):
    params = {"query": query}
    try:
        response = requests.get(FAQ_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get("answer", "No answer received.")
        else:
            return "FAQ Bot error."
    except Exception as e:
        return f"Error: {e}"

def compute_iou(box1, box2):
    # Each box is a tuple: (x1, y1, x2, y2)
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / float(area1 + area2 - inter_area)
    return iou

# Open webcam
cap = cv.VideoCapture(0)

# Variables for interaction and messages
interaction_message = ""
input_mode = False
current_input = ""
display_answer = False
answer_display_start = None
ANSWER_DISPLAY_TIME = 5  # seconds
object_messages = {}
MESSAGE_DISPLAY_TIME = 3  # seconds

# IoU threshold to consider detections as the same object
IOU_THRESHOLD = 0.5

# Optional: set a smaller size for inference for speed-up
def resize_frame(frame, width=640):
    ratio = width / frame.shape[1]
    height = int(frame.shape[0] * ratio)
    return cv.resize(frame, (width, height)), ratio

# Create a thread pool executor for concurrent inference
executor = ThreadPoolExecutor(max_workers=2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally resize frame for faster inference; adjust as needed.
    resized_frame, scale_ratio = resize_frame(frame, width=640)
    person_detected = False
    detections = []  # to store detections from both models

    # Run both inferences concurrently
    future_custom = executor.submit(model_custom, resized_frame)
    future_coco = executor.submit(model_coco, resized_frame)
    results_custom = future_custom.result()[0]
    results_coco = future_coco.result()[0]

    # Process detections from custom model
    for box in results_custom.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        label = model_custom.model.names.get(cls, "unknown")
        # Scale coordinates back to original frame size
        box_scaled = (int(x1 / scale_ratio), int(y1 / scale_ratio),
                      int(x2 / scale_ratio), int(y2 / scale_ratio))
        detections.append({
            "model": "custom",
            "box": box_scaled,
            "conf": conf,
            "label": label
        })

    # Process detections from COCO model
    for box in results_coco.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        label = model_coco.model.names.get(cls, "unknown")
        box_scaled = (int(x1 / scale_ratio), int(y1 / scale_ratio),
                      int(x2 / scale_ratio), int(y2 / scale_ratio))
        detections.append({
            "model": "coco",
            "box": box_scaled,
            "conf": conf,
            "label": label
        })

    # Merge detections: keep only the one with higher confidence for overlapping boxes
    final_detections = []
    for i, det in enumerate(detections):
        keep_det = True
        for j, other in enumerate(detections):
            if i == j:
                continue
            if det["label"].lower() == other["label"].lower():
                iou = compute_iou(det["box"], other["box"])
                if iou > IOU_THRESHOLD and other["conf"] > det["conf"]:
                    keep_det = False
                    break
        if keep_det:
            final_detections.append(det)

    # Draw final detections and trigger FAQ queries if needed
    for det in final_detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]
        label = det["label"]
        # Set color based on model source for clarity
        color = (255, 0, 0) if det["model"] == "custom" else (0, 255, 255)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Trigger FAQ query if object is not 'person'
        if label.lower() != "person":
            if label not in object_messages:
                query = f"What can you tell me about {label}?"
                answer = query_faq_chatbot(query)
                object_messages[label] = (f"Detected: {label}. {answer}", time.time())
        else:
            if label not in object_messages:
                object_messages[label] = (f"Detected: {label}", time.time())
            person_detected = True

    # Display object messages
    y_offset = 50
    current_time = time.time()
    for obj, (message, timestamp) in list(object_messages.items()):
        if current_time - timestamp < MESSAGE_DISPLAY_TIME:
            cv.putText(frame, message, (50, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30
        else:
            del object_messages[obj]

    # Manual interaction prompt if person detected
    if not input_mode and person_detected:
        cv.putText(frame, "Press 'i' to talk", (50, 400), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if input_mode:
        cv.putText(frame, "Type: " + current_input, (50, 450), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if display_answer and interaction_message:
        cv.putText(frame, interaction_message, (50, 480), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if answer_display_start and time.time() - answer_display_start > ANSWER_DISPLAY_TIME:
            display_answer = False
            interaction_message = ""

    cv.imshow("Real-Time Object Detection with Dual YOLO Models", frame)
    key = cv.waitKey(1) & 0xFF

    # Manual FAQ query handling
    if input_mode:
        if key not in [255, -1]:
            if key == 13:  # Enter key
                answer = query_faq_chatbot(current_input)
                interaction_message = answer
                display_answer = True
                answer_display_start = time.time()
                engine.say(answer)
                engine.runAndWait()
                input_mode = False
                current_input = ""
            elif key in [8, 127]:
                current_input = current_input[:-1]
            else:
                try:
                    current_input += chr(key)
                except Exception:
                    pass
    else:
        if key == ord('i') and person_detected:
            input_mode = True
            current_input = ""
    
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
executor.shutdown()
