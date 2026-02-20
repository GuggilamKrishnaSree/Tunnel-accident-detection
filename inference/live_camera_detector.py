import cv2
import numpy as np
import winsound
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

WINDOW_SIZE = 50
THRESHOLD = 0.6

cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

lstm = load_model("models/lstm_accident_detector.h5")

feature_buffer = []
alert_triggered = False


def play_alert():
    winsound.Beep(1000, 1000)


def extract_feature(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = preprocess_input(frame.astype(np.float32))
    feature = cnn.predict(frame[None, ...], verbose=0)
    return feature[0]


def generate_frames():
    global alert_triggered

    cap = cv2.VideoCapture(0)

    while True:
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
                # if not alert_triggered:
                    # play_alert()
                    # alert_triggered = True

        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")

    cap.release()
