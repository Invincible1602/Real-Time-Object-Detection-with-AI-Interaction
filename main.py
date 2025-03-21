from ultralytics import YOLO
import cv2 as cv
import requests
import time

# Load the YOLO model (using YOLOv5 pre-trained on COCO)
model = YOLO("yolov5n.pt")  

# Load COCO class names from 'coco.names'
with open("coco.names", "r") as f:
    coco_names = [line.strip() for line in f.readlines()]
# Update the internal names dictionary for proper labeling
model.model.names = {i: name for i, name in enumerate(coco_names)}

# Define the mock FAQ chatbot API endpoint URL (ensure FastAPI is running on this URL)
FAQ_API_URL = "http://localhost:8000/faq/"

def query_faq_chatbot(query):
    """Query the FAQ chatbot FastAPI endpoint."""
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

# Open a connection to the webcam (device 0)
cap = cv.VideoCapture(0)

interaction_message = ""   # To display the chatbot's answer for manual queries
input_mode = False         # Flag for when the user is typing a question
current_input = ""         # Stores the characters typed by the user
display_answer = False     # Flag to show the answer for a fixed duration

# For timing the answer display
answer_display_start = None
ANSWER_DISPLAY_TIME = 5  # seconds

# Object detection messages (stores messages for detected objects)
object_messages = {}
MESSAGE_DISPLAY_TIME = 3  # Time in seconds to display messages for detected objects

while True:
    ret, frame = cap.read()
    if not ret:
        break

    person_detected = False  # Flag to check if any person is detected
    
    # Run YOLO inference on the current frame
    results = model(frame)[0]
    
    # Process each detected object
    for box in results.boxes:
        # Extract bounding box coordinates, confidence, and class index
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        label = model.model.names.get(cls, "unknown")

        # Set bounding box color: green for person, yellow for others
        color = (0, 255, 0) if label.lower() == "person" else (0, 255, 255)
        
        # Draw the bounding box and label on the frame
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # For non-person objects, trigger chatbot interaction automatically
        if label.lower() != "person":
            if label not in object_messages:
                # Generate a query and get an answer from the FAQ chatbot
                query = f"What can you tell me about {label}?"
                answer = query_faq_chatbot(query)
                object_messages[label] = (f"Detected: {label}. {answer}", time.time())
        else:
            # For persons, just store a simple detection message if not already done
            if label not in object_messages:
                object_messages[label] = (f"Detected: {label}", time.time())
            person_detected = True

    # Display object messages
    y_offset = 50
    current_time = time.time()
    for obj, (message, timestamp) in list(object_messages.items()):
        if current_time - timestamp < MESSAGE_DISPLAY_TIME:
            cv.putText(frame, message, (50, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30  # Move text down for next message
        else:
            del object_messages[obj]  # Remove expired messages
    
    # Display chatbot interaction prompt for manual input when a person is detected
    if not input_mode and person_detected:
        cv.putText(frame, "Press 'i' to talk", (50, 400), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # If in input mode, overlay the current input text
    if input_mode:
        cv.putText(frame, "Type: " + current_input, (50, 450), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # If there's an interaction message (answer) to display from manual input
    if display_answer and interaction_message:
        cv.putText(frame, interaction_message, (50, 480), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if answer_display_start and time.time() - answer_display_start > ANSWER_DISPLAY_TIME:
            display_answer = False
            interaction_message = ""

    cv.imshow("Real-Time Object Detection with AI Interaction", frame)
    
    key = cv.waitKey(1) & 0xFF

    # If in input mode, capture keyboard input for manual FAQ queries
    if input_mode:
        if key != 255 and key != -1:  # if a key is pressed
            if key == 13:  # Enter key sends the question
                answer = query_faq_chatbot(current_input)
                interaction_message = answer
                display_answer = True
                answer_display_start = time.time()
                input_mode = False
                current_input = ""
            elif key in [8, 127]:  # Backspace key
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
  
# Release resources
cap.release()
cv.destroyAllWindows()
