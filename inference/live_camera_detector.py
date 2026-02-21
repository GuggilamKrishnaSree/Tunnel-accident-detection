import cv2
import numpy as np
import time
import os
import threading
import winsound

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from services.email_alert import send_email_alert


# ============================================================
# ⚙ CONFIGURATION
# ============================================================

WINDOW_SIZE = 50
THRESHOLD = 0.6
EMAIL_THRESHOLD = 0.65
ALERT_DURATION = 5
EMAIL_COOLDOWN = 600


# ============================================================
# 🧠 MODEL LOADING
# ============================================================

cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

lstm = load_model("models/lstm_accident_detector.h5")


# ============================================================
# 🔊 ALERT FUNCTION (Threaded Safe)
# ============================================================

def play_alert():
    winsound.Beep(1000, 1000)


# ============================================================
# 🎥 STREAM GENERATOR (NO NEW WINDOW)
# ============================================================

def generate_frames():

    feature_buffer = []
    alert_active = False
    alert_start_time = 0
    last_email_time = 0

    if not os.path.exists("static"):
        os.makedirs("static")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    print("🎥 Live Monitoring Started (Web Mode)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # ----------------------------------------------------
        # 🔹 Feature Extraction
        # ----------------------------------------------------
        resized = cv2.resize(frame, (224, 224))
        processed = preprocess_input(resized.astype(np.float32))
        feature = cnn.predict(processed[None, ...], verbose=0)[0]

        feature_buffer.append(feature)

        if len(feature_buffer) > WINDOW_SIZE:
            feature_buffer.pop(0)

        label = "Normal"
        color = (0, 255, 0)

        # ----------------------------------------------------
        # 🔹 LSTM Prediction
        # ----------------------------------------------------
        if len(feature_buffer) == WINDOW_SIZE:

            seq = np.array(feature_buffer)[None, ...]
            accident_prob = lstm.predict(seq, verbose=0)[0][0]

            # Show confidence
            cv2.putText(frame,
                        f"Confidence: {round(accident_prob,3)}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2)

            if accident_prob > THRESHOLD:

                label = "🚨 ACCIDENT DETECTED"
                color = (0, 0, 255)

                # Determine severity
                if accident_prob > 0.8:
                    severity = 3
                elif accident_prob > 0.7:
                    severity = 2
                else:
                    severity = 1

                # ------------------------------------------------
                # 🔊 ALERT CONTROL (Threaded)
                # ------------------------------------------------
                if not alert_active:
                    threading.Thread(
                        target=play_alert,
                        daemon=True
                    ).start()

                    alert_active = True
                    alert_start_time = current_time

                if alert_active and (current_time - alert_start_time > ALERT_DURATION):
                    alert_active = False

                # ------------------------------------------------
                # 📧 EMAIL CONTROL (Cooldown + Threaded)
                # ------------------------------------------------
                if accident_prob > EMAIL_THRESHOLD and \
                   (current_time - last_email_time > EMAIL_COOLDOWN):

                    snapshot_path = "static/live_snapshot.jpg"
                    cv2.imwrite(snapshot_path, frame)

                    print("📧 Sending email alert...")

                    threading.Thread(
                        target=send_email_alert,
                        args=(severity, accident_prob, snapshot_path),
                        daemon=True
                    ).start()

                    last_email_time = current_time

            else:
                alert_active = False

        # ----------------------------------------------------
        # 🔹 Draw Status
        # ----------------------------------------------------
        cv2.putText(frame,
                    label,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2)

        # ----------------------------------------------------
        # 🔹 Encode for Flask (NO imshow here)
        # ----------------------------------------------------
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")

    cap.release()