import cv2
import av
import mediapipe as mp
import streamlit as st
import numpy as np
import logging # Import logging to handle deepface warnings
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from deepface import DeepFace # The powerful facial analysis library

# ==========================================
# 1. Global Setup and Logging Configuration
# ==========================================
# Suppress the lengthy warnings from deepface/tensorflow to keep the console clean.
logging.getLogger("deepface").setLevel(logging.ERROR)

# ==========================================
# 2. Set up MediaPipe Hand Tracking Solutions
# ==========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands model for static complex images or video streams
# These confidence settings are crucial for a responsive interface.
hands = mp_hands.Hands(
    model_complexity=1, # 1 is a good balance of speed vs. stability
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)

# ==========================================
# 3. Streamlit Page Configuration
# ==========================================
# Set up a multi-column and centered layout for better presentation.
st.set_page_config(page_title="Human Signal AI", page_icon="✋😊", layout="centered")

st.title("Human Signal AI: Hands & Face Identification ✋🌏")
st.markdown("---")
st.write("**Using AI to recognize and translate your hand signals *and* facial expressions in real-time.**")
st.write("Grant webcam access, hold your hand clearly in the frame, and see the AI analyze both gesture and emotion.")

# We create two columns to provide instructions for a better user experience.
col1, col2 = st.columns(2)
with col1:
    st.info("**Hand Gestures to Try:** Thumbs Up, Fist, Open Palm, Rock/Horns Sign, Spiderman/Shaka, Victory.")
with col2:
    st.info("**Facial Emotions to Try:** Angry, Happy, Sad, Surprise, Neutral.")

# ==========================================
# 4. Advanced Video Processing Class (Updated)
# ==========================================
# This class contains all logic for processing each individual frame from the webcam.
class MultiSignalRecognitionProcessor:
    def __init__(self):
        # Optional: Setup a basic face detector for optimization to prevent full deepface
        # calls when no face is likely detected. Let's stick with calling DeepFace for simplicity first.
        # DeepFace can find faces but having a simple detector helps.
        pass

    def _get_finger_states(self, landmarks):
        """
        Determines if individual fingers are open (1) or closed (0) 
        based on standard joint analysis rules. Landmark indices from standard hand models.
        """
        finger_states = []

        # Hand tips: [Index(8), Middle(12), Ring(16), Pinky(20)]
        tips = [8, 12, 16, 20]
        # PIP (Proximal Interphalangeal) joints for respective tips
        pips = [6, 10, 14, 18]

        # Standard logic for the four regular fingers: a tip is open if it's vertically above its PIP.
        for i in range(len(tips)):
            if landmarks[tips[i]].y < landmarks[pips[i]].y:
                finger_states.append(1) # Up / Open
            else:
                finger_states.append(0) # Down / Closed

        # Specialized logic for the Thumb. It's usually folded horizontally.
        thumb_tip = landmarks[4]
        thumb_is_open = False
        
        # Compare horizontal distance from Tip (4) to MCP joint (5) against MCP proximal joint (3).
        # We look for simple horizontal extension.
        if abs(thumb_tip.x - landmarks[5].x) > abs(landmarks[3].x - landmarks[5].x):
            thumb_is_open = True

        if thumb_is_open:
            finger_states.insert(0, 1) # Insert Thumb state at start of list
        else:
            finger_states.insert(0, 0)
        
        return finger_states

    def _classify_gesture(self, finger_states):
        """
        Maps a 5-element binary list (representing [Thumb, Index, Middle, Ring, Pinky])
        to a common gesture name or signal.
        """
        # Common Gestures definitions:
        if finger_states == [0, 0, 0, 0, 0]:
            return "Fist / Closed Hand"
        elif finger_states == [0, 1, 0, 0, 0]:
            return "Point / Selection"
        elif finger_states == [0, 1, 1, 0, 0]:
            return "Victory / Peace Sign"
        elif finger_states == [1, 1, 1, 1, 1]:
            return "Open Palm / Greeting"
        elif finger_states == [1, 0, 0, 0, 0]:
            return "Thumbs Up / OK (Simple)"
        elif finger_states == [0, 1, 0, 0, 1]:
            return "Rock / Horns Sign"
        elif finger_states == [1, 1, 0, 0, 1]:
            return "Spiderman / Shaka Sign"
        elif finger_states == [1, 0, 0, 0, 1]: # Only thumb and pinky up
            return "Call Me Gesture"
        
        return "Unknown Signal"

    def _classify_face_expression(self, img_rgb):
        """
        Uses DeepFace to analyze facial expressions and return the dominant emotion.
        Includes error handling and optimization for faster real-time processing.
        """
        try:
            # OPTIMIZATION: Reduce image size for deepface to run significantly faster on standard CPUs.
            analysis_frame = cv2.resize(img_rgb, (0, 0), fx=0.5, fy=0.5)
            
            # Use DeepFace with focused analysis on emotion and non-enforced detection.
            # Choosing detector_backend='opencv' is slower but doesn't require extra C++ setups.
            results = DeepFace.analyze(analysis_frame, 
                                        actions=['emotion'], 
                                        enforce_detection=False, # Don't crash if no face is found
                                        detector_backend='opencv') 
            
            if results and len(results) > 0:
                # DeepFace returns a list of dictionaries for each detected face.
                # Use a specific list of emotion mapping to get dominant, user-friendly labels.
                # A complete list of emotions is needed for comprehensive analysis.
                dominant_emotion = results[0]['dominant_emotion']
                emotion_mapping = {
                    'angry': 'Angry',
                    'disgust': 'Disgust',
                    'fear': 'Fear',
                    'happy': 'Happy',
                    'sad': 'Sad',
                    'surprise': 'Surprise',
                    'neutral': 'Neutral'
                }
                return emotion_mapping.get(dominant_emotion, dominant_emotion.capitalize())
            return "No Face Detected"
        except Exception as e:
            # Fallback for complex errors from DeepFace models.
            return f"Face analysis error: {str(e)}"

    def recv(self, frame):
        """This function processes every incoming video frame from the user's webcam."""
        
        # 1. Convert the incoming WebRTC frame to an OpenCV-compatible numpy array
        img = frame.to_ndarray(format="bgr24")

        # 2. Convert colorspace for MediaPipe and deepface processing (BGR to RGB)
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ==========================================
        # SIGNAL 1: Hand Gesture Analysis (Existing)
        # ==========================================
        # Process the image to find hand landmarks first.
        results = hands.process(img_rgb)
        gesture_text = "No Hand Detected"
        img.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # First, draw standard connections and joints ("skeleton") on the hand.
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Then analyze standard finger states using standard finger index list.
                finger_states = self._get_finger_states(hand_landmarks.landmark)
                gesture_text = self._classify_gesture(finger_states)

        # ==========================================
        # SIGNAL 2: Facial Expression Analysis (NEW Update)
        # ==========================================
        face_emotion_text = self._classify_face_expression(img_rgb)

        # ==========================================
        # FINAL DISPLAY: Output Overlays (Updated and Enhanced)
        # ==========================================
        # The coordinate system is (0,0) at the TOP LEFT. 
        # For a natural webcam experience for the user, we have to flip horizontally.
        img = cv2.flip(img, 1) # Flip first before drawing text so the text isn't mirrored.

        # Text visual properties
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0
        font_thickness = 2
        
        # Color palettes - high contrast for readability
        prefix_color = (255, 255, 255) # White labels
        gesture_color = (0, 255, 255) # Cyan hand gesture
        emotion_color = (255, 255, 0) # Yellow face emotion

        # Define multi-line layout logic for clear presentation.
        margin = 20
        # Determine visual heights for line spacing based on font size.
        (label_w, text_h), baseline = cv2.getTextSize("Hand: ", font_face, font_scale, font_thickness)
        # Calculate standard line spacing.
        line_spacing = text_h + baseline + 15
        
        # Horizontal layout logic - find max text width to draw standard background rect.
        gesture_label = "Hand Signal: "
        face_label = "Facial Emotion: "
        
        combined_gesture_text = f"{gesture_label}{gesture_text}"
        combined_face_text = f"{face_label}{face_emotion_text}"

        (g_w, g_h), g_b = cv2.getTextSize(combined_gesture_text, font_face, font_scale, font_thickness)
        (f_w, f_h), f_b = cv2.getTextSize(combined_face_text, font_face, font_scale, font_thickness)
        
        max_text_width = max(g_w, f_w)
        # Total rectangle height for multi-line block with padding.
        total_block_height = line_spacing * 2 - 10

        # Translucent Background Rectangle
        # We need alpha blending for a clear translucent overlay.
        overlay = img.copy()
        cv2.rectangle(overlay, (margin - 10, margin), (margin + max_text_width + 10, margin + total_block_height), (0, 0, 0), -1)
        # Blend translucency.
        alpha = 0.55 # Opacity
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Draw hand signal line (Top left within multi-line block).
        text_origin_g = (margin, margin + text_h + 10)
        # Part 1: prefix label
        cv2.putText(img, gesture_label, text_origin_g, font_face, font_scale, prefix_color, font_thickness, cv2.LINE_AA)
        # Part 2: value (with precise x offset).
        (prefix_w_g, _), _ = cv2.getTextSize(gesture_label, font_face, font_scale, font_thickness)
        value_origin_g = (text_origin_g[0] + prefix_w_g, text_origin_g[1])
        cv2.putText(img, gesture_text, value_origin_g, font_face, font_scale, gesture_color, font_thickness, cv2.LINE_AA)

        # Draw face emotion line (below the hand signal line).
        text_origin_f = (margin, text_origin_g[1] + line_spacing - 10)
        # Part 1: prefix label
        cv2.putText(img, face_label, text_origin_f, font_face, font_scale, prefix_color, font_thickness, cv2.LINE_AA)
        # Part 2: value (with precise x offset).
        (prefix_w_f, _), _ = cv2.getTextSize(face_label, font_face, font_scale, font_thickness)
        value_origin_f = (text_origin_f[0] + prefix_w_f, text_origin_f[1])
        cv2.putText(img, face_emotion_text, value_origin_f, font_face, font_scale, emotion_color, font_thickness, cv2.LINE_AA)

        # 8. Return the final, processed frame back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. Configure and Start WebRTC
# ==========================================
# STUN servers allow your browser to communicate the video feed through firewalls.
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# Start the webcam streamer component with our multi-processing class.
webrtc_streamer(
    key="human-signal-tracker",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    # Standard webcam settings; disable audio for focus on video processing.
    media_stream_constraints={"video": True, "audio": False},
    # Use standard Python multi-threading with asynchronous processing for a smooth interface.
    video_processor_factory=MultiSignalRecognitionProcessor,
    async_processing=True,
)

st.markdown("---")
st.write("**Technical Stack:** Standard landmark indices for hand structure and standard deep learning models for comprehensive emotion identification are utilized.")