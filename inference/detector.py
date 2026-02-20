import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

WINDOW_SIZE = 50
THRESHOLD = 0.5
FPS = 10  

cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

lstm = load_model("models/lstm_accident_detector.h5")


def extract_feature(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = preprocess_input(frame.astype(np.float32))
    feature = cnn.predict(frame[None, ...], verbose=0)
    return feature[0]


def detect_accident_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    feature_buffer = []

    events = []

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        feature = extract_feature(frame)
        feature_buffer.append(feature)

        if len(feature_buffer) > WINDOW_SIZE:
            feature_buffer.pop(0)

        if len(feature_buffer) == WINDOW_SIZE:
            seq = np.array(feature_buffer)[None, ...]
            prob = lstm.predict(seq, verbose=0)[0][0]

            if prob > THRESHOLD:
                confidence = float(prob)
                timestamp = frame_idx/FPS
                
                frame_filename = f"accident_{timestamp:.1f}.jpg"
                frame_path = os.path.join("static/frames",frame_filename)
                
                cv2.imwrite(frame_path, frame)
                events.append({
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "frame": frame_filename
                })

        frame_idx += 1

    cap.release()

    return {
        "accident_detected" : len(events) > 0,
        "events"  : events
    }
