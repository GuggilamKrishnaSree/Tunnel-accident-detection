import numpy as np
from sklearn.model_selection import train_test_split
from models.lstm_model import build_lstm_model
from data_loader.anticipation_sequence_loader import load_anticipation_sequences
from preprocessing.parse_ccd_annotations import parse_crash_1500

WINDOW_SIZE = 16
EPOCHS = 25
BATCH_SIZE = 16

# Load annotations
crash_dict = parse_crash_1500("data/raw_videos/CCD/Crash-1500.txt")

# Load sequences
X, y = load_anticipation_sequences(
    crash_features_dir="data/processed/features/crash-1500",
    normal_features_dir="data/processed/features/normal",
    crash_annotation_dict=crash_dict
)

print("Sequences:", X.shape, "Labels:", y.shape)

# Train / Val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Build model
model = build_lstm_model(timesteps=WINDOW_SIZE)

model.summary()

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Save model
model.save("models/anticipation_lstm.h5")
print("✅ Model saved")
