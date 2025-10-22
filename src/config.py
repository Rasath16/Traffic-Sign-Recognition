"""Configuration settings."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Model paths
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"