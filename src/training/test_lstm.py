import sys
import os

sys.path.append(os.path.abspath("src"))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from tensorflow.keras.models import load_model

from data_loader.sequence_loader import load_sequences

# ---------------- CONFIG ----------------
TIMESTEPS = 50
MODEL_PATH = "models/lstm_accident_detector.h5"
THRESHOLD = 0.5
# ----------------------------------------

# 1️⃣ Load Test Data
X, y = load_sequences(max_timesteps=TIMESTEPS)

print("Dataset shape:", X.shape)

# ⚠ IMPORTANT:
# If this loads full dataset, ideally you should
# create a separate test split.
# If not already split earlier, we split here:

from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Test shape:", X_test.shape)

# 2️⃣ Load Trained Model
model = load_model(MODEL_PATH)

# 3️⃣ Evaluate using built-in Keras
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("\n✅ Keras Test Accuracy:", accuracy)

# 4️⃣ Predict probabilities
y_pred_prob = model.predict(X_test)

# Convert to binary predictions
y_pred = (y_pred_prob > THRESHOLD).astype(int)

# 5️⃣ Detailed Metrics
print("\n📊 Accuracy (sklearn):",
      accuracy_score(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6️⃣ ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("\n📊 ROC-AUC Score:", roc_auc)