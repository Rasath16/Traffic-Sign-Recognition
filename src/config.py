"""
Configuration file for Traffic Sign Recognition project
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

# GTSRB Dataset paths (based on your structure)
TRAIN_DIR = os.path.join(RAW_DATA_DIR, 'Train')
TEST_DIR = os.path.join(RAW_DATA_DIR, 'Test')
META_DIR = os.path.join(RAW_DATA_DIR, 'Meta')

TRAIN_CSV = os.path.join(RAW_DATA_DIR, 'Train.csv')
TEST_CSV = os.path.join(RAW_DATA_DIR, 'Test.csv')
META_CSV = os.path.join(RAW_DATA_DIR, 'Meta.csv')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Image parameters
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3

# Model parameters
NUM_CLASSES = 43  # GTSRB has 43 traffic sign classes
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Data split
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.15

# Class names for GTSRB
CLASS_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}