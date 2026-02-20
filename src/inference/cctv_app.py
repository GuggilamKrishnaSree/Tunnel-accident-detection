# import cv2
# from detect_accident_video import detect_from_frame as detect_accident,play_alert

# VIDEO_SOURCE = 0  # 0 = webcam, or path to video file

# cap = cap = cv2.VideoCapture(0)
# alert_triggered = False

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run detection
#     accident_prob = detect_accident(frame)

#     # Display CCTV feed
#     cv2.putText(frame, "CCTV MONITOR", (20, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     if accident_prob > 0.5:
#         cv2.putText(frame, "🚨 ACCIDENT DETECTED", (50, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

#         # if not alert_triggered:
#         #     play_alert()
#         #     alert_triggered = True
#         play_alert()

#     cv2.imshow("Tunnel CCTV Accident Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import time
import os

from detect_accident_video import detect_from_frame as detect_accident, play_alert
from services.email_alert import send_email_alert


VIDEO_SOURCE = 0  # webcam or video path

cap = cv2.VideoCapture(VIDEO_SOURCE)

# -------------------------
# CONFIGURATION
# -------------------------

ALERT_THRESHOLD = 0.5        # show alert text
EMAIL_THRESHOLD = 0.6        # send email if above this
ALERT_DURATION = 5           # seconds siren active
EMAIL_COOLDOWN = 600          # seconds between emails

# -------------------------
# STATE VARIABLES
# -------------------------

alert_active = False
alert_start_time = 0
last_email_time = 0

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")

print("🎥 Live Tunnel CCTV Monitoring Started... Press 'q' to quit.")

# -------------------------
# MAIN LOOP
# -------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    accident_prob = detect_accident(frame)

    # Display CCTV header
    cv2.putText(frame, "CCTV MONITOR",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    current_time = time.time()

    # -------------------------
    # ACCIDENT DETECTION
    # -------------------------
    if accident_prob > ALERT_THRESHOLD:

        # Display alert text
        cv2.putText(frame,
                    "ACCIDENT DETECTED",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3)

        # Determine severity dynamically
        if accident_prob > 0.8:
            severity = 3
        elif accident_prob > 0.65:
            severity = 2
        else:
            severity = 1

        # -------------------------
        # 🔊 ALERT CONTROL
        # -------------------------
        if not alert_active:
            play_alert()
            alert_active = True
            alert_start_time = current_time

        # Stop alert after duration
        if alert_active and (current_time - alert_start_time > ALERT_DURATION):
            alert_active = False

        # -------------------------
        # 📧 EMAIL CONTROL (Cooldown Protected)
        # -------------------------
        if accident_prob > EMAIL_THRESHOLD and \
           (current_time - last_email_time > EMAIL_COOLDOWN):

            snapshot_path = "static/live_snapshot.jpg"
            cv2.imwrite(snapshot_path, frame)
            send_email_alert(
                severity=severity,
                confidence=accident_prob,
                snapshot_path=snapshot_path
            )

            print("📧 Email alert sent successfully.")
            last_email_time = current_time

    else:
        # Reset alert if no accident
        alert_active = False

    # Show window
    cv2.imshow("Tunnel CCTV Accident Detection", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("🛑 Monitoring Stopped.")
