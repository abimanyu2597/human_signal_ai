# Human Signal AI: Hands & Face Identification ✋😊

A real-time AI-powered web application built with Streamlit that simultaneously recognizes hand gestures and facial expressions using your webcam.

This project combines Google MediaPipe for advanced hand tracking with DeepFace for facial emotion recognition.

---

## 🚀 Features

### ✋ Real-Time Hand Tracking
- Detects 21 hand landmarks using MediaPipe
- Draws landmarks live on the video feed

### 🖐 Gesture Classification
Rule-based detection of common gestures:
- 👍 Thumbs Up
- ✊ Fist
- ✌️ Victory / Peace
- 🖐 Open Palm
- 🤟 Spiderman / Shaka

### 😊 Facial Expression Recognition
Emotion detection powered by DeepFace:
- 😊 Happy
- 😠 Angry
- 😢 Sad
- 😮 Surprise
- 😐 Neutral

### ⚡ Performance Optimization
- Image resizing for speed
- Async processing
- macOS compatibility fixes
- TensorFlow 2.16+ support

---

## 🛠 Technical Stack

- Frontend & Server: Streamlit  
- Video Streaming: Streamlit-WebRTC  
- Hand Detection: Google MediaPipe (Hands)  
- Face Detection: DeepFace (RetinaFace backend)  
- Deep Learning: TensorFlow (v2.16+) + tf-keras  
- Image Processing: OpenCV (Headless)

---

## 📦 Installation & Setup

### Requirements
- Python 3.9+
- Webcam
- macOS / Windows / Linux

---

### 1️⃣ Initialize Project

```bash
mkdir human-signal-ai
cd human-signal-ai


Create Virtual Environment
python -m venv venv
source venv/bin/activate

Install Dependencies
pip install streamlit streamlit-webrtc mediapipe opencv-python-headless av deepface tensorflow tf-keras

The tf-keras package fixes compatibility issues with TensorFlow 2.16+.


Save Your Code

Ensure your application code is saved as:

app.py

inside the human-signal-ai directory.

Run the Application
streamlit run app.py

Open your browser at:

http://localhost:8501

Click Start, allow webcam access, and test gestures and emotions in real-time.

🧩 Troubleshooting
TensorFlow 2.16+ Error

If you encounter:

ValueError: You have tensorflow X.X.X and this requires tf-keras package...

Run:

pip install tf-keras

inside your activated virtual environment.

First Run Note

On first launch:

DeepFace downloads pre-trained models

This may take several minutes

Subsequent runs will be faster

👨‍💻 Author

Built by Raja Abimanyu N
Data Scientist | AI & Applied ML




