from flask import Flask, render_template, request, Response, send_from_directory
import os
from inference.detector import detect_accident_in_video
from inference.live_camera_detector import generate_frames
from utils.video_utils import normalize_video
from werkzeug.utils import secure_filename


app = Flask(__name__)

RAW_FOLDER = "uploads/raw"
PROCESSED_FOLDER = "uploads/processed"
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        video = request.files["video"]
        if not video or video.filename == "":
            return render_template("upload.html", error="No file selected")
        
        filename = secure_filename(video.filename)
        
        raw_path = os.path.join(RAW_FOLDER, filename)
        processed_path = os.path.join(PROCESSED_FOLDER,filename)
        
        video.save(raw_path)
        normalize_video(raw_path, processed_path)
        result = detect_accident_in_video(processed_path)

        return render_template(
            "upload.html",
            result = result,
            video_filename = filename
        )

    return render_template("upload.html")

@app.route("/live")
def live():
    return render_template("live.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    
@app.route("/uploads/processed/<filename>")
def uploaded_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename,   mimetype='video/mp4')

if __name__ == "__main__":
    app.run(debug=True)
