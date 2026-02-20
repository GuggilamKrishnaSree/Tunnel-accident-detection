import cv2
import time
import os
import threading

from src.inference.detect_accident_video import detect_from_frame as detect_accident, play_alert
from services.email_alert import send_email_alert


def generate_frames():
    VIDEO_SOURCE = 0  # 0 for webcam, or provide video file path
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video source.")
        exit()
        
    ALERT_THRESHOLD = 0.5      # Show alert text
    EMAIL_THRESHOLD = 0.6      # Send email if above this
    ALERT_DURATION = 5         # Seconds siren active
    EMAIL_COOLDOWN = 600       # Seconds between emails (10 min)
    
    alert_active = False
    alert_start_time = 0
    last_email_time = 0
    
    # Ensure static folder exists
    if not os.path.exists("static"):
        os.makedirs("static")

    print("🎥 Live Tunnel CCTV Monitoring Started...")
    print("Press 'q' to quit.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Run detection
        accident_prob = detect_accident(frame)

        # Safety check
        if accident_prob is None:
            accident_prob = 0.0

        # Display CCTV header
        cv2.putText(frame, "CCTV MONITOR",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

        current_time = time.time()

        if accident_prob > ALERT_THRESHOLD:
            # Display detection text
            cv2.putText(frame,
                    "ACCIDENT DETECTED",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3)

            # Determine severity
            if accident_prob > 0.8:
                severity = 3
            elif accident_prob > 0.65:
                severity = 2
            else:
                severity = 1

            # ----------------------------------------------------
            # 🔊 ALERT CONTROL (Threaded, Non-Blocking)
            # ----------------------------------------------------
            if not alert_active:
                print("🔊 Playing alert sound...")
                threading.Thread(
                target=play_alert,
                daemon=True
                ).start()

                alert_active = True
                alert_start_time = current_time

            # Stop alert flag after duration
            if alert_active and (current_time - alert_start_time > ALERT_DURATION):
                alert_active = False

            # ----------------------------------------------------
            # 📧 EMAIL CONTROL (Cooldown Protected, Threaded)
            # ----------------------------------------------------
            if accident_prob > EMAIL_THRESHOLD and \
            (current_time - last_email_time > EMAIL_COOLDOWN):

                print("📧 Sending email alert...")

                snapshot_path = "static/live_snapshot.jpg"
                cv2.imwrite(snapshot_path, frame)

                threading.Thread(
                target=send_email_alert,
                args=(severity, accident_prob, snapshot_path),
                daemon=True
                ).start()

                last_email_time = current_time

        else:
            alert_active = False

        # Show probability on screen (for debugging)
        cv2.putText(frame,
                f"Confidence: {round(accident_prob, 3)}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

        # Display window
        cv2.imshow("Tunnel CCTV Accident Detection", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # ============================================================
    # 🛑 CLEANUP
    # ============================================================

    cap.release()
    cv2.destroyAllWindows()

    print("🛑 Monitoring Stopped.")