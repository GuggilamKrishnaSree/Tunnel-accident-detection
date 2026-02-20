import smtplib
from email.message import EmailMessage
import ssl
from datetime import datetime
import os

# 🔐 Replace with your credentials
SENDER_EMAIL = "vasantha70938@gmail.com"
SENDER_PASSWORD = "kudgzzgwgubsspyq"
RECEIVER_EMAIL = "vasantha70938@gmail.com"

def send_email_alert(severity, confidence, snapshot_path=None):

    msg = EmailMessage()
    msg["Subject"] = f"🚨 Tunnel Accident Alert - Severity {severity}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    body = f"""
    🚨 ACCIDENT DETECTED 🚨

    Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Severity Level: {severity}
    Confidence Score: {round(confidence, 3)}

    Immediate attention may be required.
    """

    msg.set_content(body)

    # Attach snapshot if exists
    if snapshot_path and os.path.exists(snapshot_path):
        with open(snapshot_path, "rb") as f:
            img_data = f.read()
            msg.add_attachment(
                img_data,
                maintype="image",
                subtype="jpeg",
                filename="live_accident_snapshot.jpg"
            )

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
