"""A simple custom CNN architecture (placeholder)."""
from tensorflow.keras import layers, models


def build_simple_cnn(input_shape=(64, 64, 3), num_classes=43):
    model = models.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model
