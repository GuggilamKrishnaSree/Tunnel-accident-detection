import cv2
import os
from pathlib import Path

# Paths
RAW_VIDEO_DIR = "data/raw_videos/CCD"
OUTPUT_FRAME_DIR = "data/processed_frames/CCD"

VIDEO_CATEGORIES = ["Normal", "Crash-1500"]
IMG_SIZE = (224, 224)

def extract_frames_from_video(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, IMG_SIZE)
        frame_name = f"frame_{frame_idx:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)

        frame_idx += 1

    cap.release()

def main():
    for category in VIDEO_CATEGORIES:
        input_category_path = Path(RAW_VIDEO_DIR) / category
        output_category_path = Path(OUTPUT_FRAME_DIR) / category

        for video_file in sorted(input_category_path.glob("*.mp4")):
            video_id = video_file.stem
            output_video_dir = output_category_path / video_id

            print(f"Extracting frames from {video_file}")
            extract_frames_from_video(video_file, output_video_dir)

if __name__ == "__main__":
    main()
