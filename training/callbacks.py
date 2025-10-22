"""Custom training callbacks (placeholders)."""
from tensorflow.keras.callbacks import Callback


class SimpleLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch} finished. Logs: {logs}")
