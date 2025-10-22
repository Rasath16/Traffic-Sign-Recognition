"""Grad-CAM utilities placeholder."""
import numpy as np


def compute_gradcam(model, image, layer_name: str = None):
    """Return dummy heatmap."""
    h, w = image.size
    return np.zeros((h, w))
