import os
import numpy as np

WINDOW_SIZE = 16
STRIDE = 5


def load_anticipation_sequences(
    crash_features_dir,
    normal_features_dir,
    crash_annotation_dict
):
    """
    crash_features_dir: path to crash .npy files
    normal_features_dir: path to normal .npy files
    crash_annotation_dict: { video_id: accident_start_frame }
    """

    X, y = [], []

    # --------------------
    # 1️⃣ Load CRASH videos
    # --------------------
    for video_id, accident_frame in crash_annotation_dict.items():
        feature_path = os.path.join(crash_features_dir, f"{video_id}.npy")

        if not os.path.exists(feature_path):
            continue

        features = np.load(feature_path)  # (50, 1280)
        num_frames = features.shape[0]

        for end in range(WINDOW_SIZE, num_frames, STRIDE):
            seq = features[end - WINDOW_SIZE:end]

            # Anticipation label
            label = 0 if end < accident_frame else 1

            X.append(seq)
            y.append(label)

    # --------------------
    # 2️⃣ Load NORMAL videos
    # --------------------
    for file in os.listdir(normal_features_dir):
        if not file.endswith(".npy"):
            continue

        feature_path = os.path.join(normal_features_dir, file)
        features = np.load(feature_path)
        num_frames = features.shape[0]

        for end in range(WINDOW_SIZE, num_frames, STRIDE):
            seq = features[end - WINDOW_SIZE:end]

            X.append(seq)
            y.append(0)  # always normal

    return np.array(X), np.array(y)
