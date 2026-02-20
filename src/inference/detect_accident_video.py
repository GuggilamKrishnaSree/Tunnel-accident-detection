import cv2
import winsound
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

WINDOW_SIZE = 50
THRESHOLD = 0.6

# Load models
cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)
lstm = load_model("models/lstm_accident_detector.h5")

def play_alert():
    duration = 1000  # milliseconds
    frequency = 1000  # Hz
    winsound.Beep(frequency, duration)

def extract_feature(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = preprocess_input(frame.astype(np.float32))
    feature = cnn.predict(frame[None, ...], verbose=0)
    return feature[0]

def detect_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    feature_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        feature = extract_feature(frame)
        feature_buffer.append(feature)

        if len(feature_buffer) > WINDOW_SIZE:
            feature_buffer.pop(0)

        label = "Normal"
        color = (0, 255, 0)

        if len(feature_buffer) == WINDOW_SIZE:
            seq = np.array(feature_buffer)[None, ...]
            prob = lstm.predict(seq, verbose=0)[0][0]

            if prob > THRESHOLD:
                label = "🚨 ACCIDENT DETECTED"
                color = (0, 0, 255)
                play_alert()

        cv2.putText(
            frame, label, (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

        cv2.imshow("Accident Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

feature_buffer = [] 
def detect_from_frame(frame):
    global feature_buffer

    feature = extract_feature(frame)
    feature_buffer.append(feature)

    if len(feature_buffer) > WINDOW_SIZE:
        feature_buffer.pop(0)

    if len(feature_buffer) < WINDOW_SIZE:
        return 0.0  # not enough context yet

    seq = np.array(feature_buffer)[None, ...]
    prob = lstm.predict(seq, verbose=0)[0][0]
    return prob
