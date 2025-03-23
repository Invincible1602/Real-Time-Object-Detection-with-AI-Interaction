# Real-Time-Object-Detection-with-AI-Interaction

This project integrates real-time object detection using dual YOLO models with a FastAPI-based FAQ chatbot. The system detects objects from both custom and COCO datasets and provides interactive responses based on detected objects.

## Features

- **Dual YOLO Models**: Utilizes both custom-trained YOLO model (best.pt) and COCO-pretrained YOLO models (yolov5n.pt) for comprehensive object detection.
- **FAQ Chatbot**: Integrates a FastAPI-based chatbot that provides information about detected objects.
- **Real-Time Interaction**: Detects objects in real-time using webcam input and offers interactive responses.
- **Text-to-Speech (TTS)**: Converts chatbot responses to speech for an enhanced user experience.

## Requirements

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/Real-Time-Object-Detection-with-AI-Interaction.git
   cd Real-Time-Object-Detection-with-AI-Interaction
   ```

2. **Running the Project**:

1. **Start the FAQ Chatbot API**:
   - If running the chatbot locally, start the FastAPI server using:
```bash
cd chatbot
python main.py 
```
The chatbot will be accessible at: http://localhost:8000/faq/


2. **Run the Object Detection Script**: 
   - Open a new terminal (with the virtual environment activated) and run:
bash
```bash
python main3.py
```
The script will open a window with the webcam feed, perform real-time object detection, and interact with the FAQ chatbot based on the detected objects.

3. **Interacting with the System**
- Automatic FAQ Query: For non-person objects detected, the system automatically sends queries to the FAQ chatbot and displays the responses.

- Manual Interaction: When a person is detected, the system displays a prompt ("Press 'i' to talk"). Press 'i' to enter input mode, type your question, and press Enter to receive a response.

- Exiting the Application: Press 'q' to quit the object detection application.



