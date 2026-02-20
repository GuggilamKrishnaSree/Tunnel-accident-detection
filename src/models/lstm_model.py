import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

def build_lstm_model(
    timesteps,
    feature_dim=1280,
    num_classes=1
):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(timesteps, feature_dim)),

        LSTM(256, return_sequences=True),
        Dropout(0.3),

        LSTM(128),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.2),

        Dense(num_classes, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model