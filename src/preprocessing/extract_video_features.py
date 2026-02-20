import os
from feature_extractor import extract_features

FRAMES_ROOT = "data/processed/frames/CCD"
FEATURES_ROOT = "data/processed/features"

for split in ["Crash-1500", "Normal"]:
    split_frames_dir = os.path.join(FRAMES_ROOT, split)
    split_features_dir = os.path.join(FEATURES_ROOT, split)

    os.makedirs(split_features_dir, exist_ok=True)

    for video_id in sorted(os.listdir(split_frames_dir)):
        frames_dir = os.path.join(split_frames_dir, video_id)

        if not os.path.isdir(frames_dir):
            continue

        save_path = os.path.join(
            split_features_dir,
            f"{video_id}.npy"
        )

        print(f"[INFO] Extracting features for {split}/{video_id}")
        extract_features(frames_dir, save_path)
