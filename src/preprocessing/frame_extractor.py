import cv2
import os

def extract_frames(video_path, output_dir, fps=5):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(video_fps // fps, 1)

    frame_id = 0
    saved_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(
                os.path.join(output_dir, f"{saved_id:04d}.jpg"),
                frame
            )
            saved_id += 1

        frame_id += 1

    cap.release()
