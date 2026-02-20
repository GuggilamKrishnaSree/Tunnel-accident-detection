import cv2
import numpy as np
import random

def apply_gamma(image, gamma=0.7):
    if image is None:
        raise ValueError("Input image is None")

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)

def reduce_brightness(image, factor=0.4):
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def reduce_contrast(image, alpha=0.6):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def add_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def simulate_tunnel_environment(image, training=True):
    if image is None:
        raise ValueError("Image not loaded properly")

    if not training:
        return image  # 🚨 IMPORTANT for validation/testing

    if random.random() < 0.8:
        image = reduce_brightness(image)

    if random.random() < 0.7:
        image = reduce_contrast(image)

    if random.random() < 0.5:
        image = apply_gamma(image)

    if random.random() < 0.4:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    if random.random() < 0.3:
        image = add_gaussian_noise(image)

    return image
