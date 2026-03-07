import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

FEATURE_ROOT = "src/data/processed/features"

def load_sequences(max_timesteps=50):
    X = []
    y = []

    for label_dir, label in [("Crash-1500", 1), ("Normal", 0)]:
        dir_path = FEATURE_ROOT+"/"+label_dir
        print(dir_path)

        for file in sorted(os.listdir(dir_path)):
            if not file.endswith(".npy"):
                continue

            path = os.path.join(dir_path, file)
            features = np.load(path)  # (Ti, 1024)

            # truncate if too long
            features = features[:max_timesteps]

            X.append(features)
            y.append(label)

    # Pad sequences
    X = pad_sequences(
        X,
        maxlen=max_timesteps,
        dtype="float32",
        padding="post",
        truncating="post"
    )

    y = np.array(y)

    return X, y

# Run this code once
# from sequence_loader import load_sequences

# X, y = load_sequences(max_timesteps=50)

# print(X.shape)  # (num_videos, 50, 1024)
# print(y.shape)  # (num_videos,)
# print("Crash:", sum(y), "Normal:", len(y) - sum(y))
