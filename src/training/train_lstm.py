import sys
import os

sys.path.append(os.path.abspath("src"))

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_loader.sequence_loader import load_sequences
from models.lstm_model import build_lstm_model

# ---------------- CONFIG ----------------
TIMESTEPS = 50
BATCH_SIZE = 8
EPOCHS = 30
MODEL_PATH = "models/lstm_accident_detector.h5"
# ----------------------------------------

# Load data
X, y = load_sequences(max_timesteps=TIMESTEPS)

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Train:", X_train.shape, "Val:", X_val.shape)

# Build model
model = build_lstm_model(
    timesteps=TIMESTEPS,
    feature_dim=1280,
    num_classes=1
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# Save final model
model.save(MODEL_PATH)
print("✅ Model saved to", MODEL_PATH)


