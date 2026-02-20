import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tunnel_simulation import simulate_tunnel_environment

# Load pretrained CNN
model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

def extract_features(frame_dir, save_path):
    features = []

    for img_name in sorted(os.listdir(frame_dir)):
        img_path = os.path.join(frame_dir, img_name)

        img = cv2.imread(img_path)
        img = simulate_tunnel_environment(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        feature = model.predict(img, verbose=0)
        features.append(feature.squeeze())

    features = np.array(features)
    np.save(save_path, features)
