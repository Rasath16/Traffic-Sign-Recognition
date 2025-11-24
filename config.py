import os

# PATHS
DATA_DIR = "data/GTSRB"
TRAIN_CSV = os.path.join(DATA_DIR, "Train.csv")
TEST_CSV = os.path.join(DATA_DIR, "Test.csv")
META_CSV = os.path.join(DATA_DIR, "Meta.csv")

# Model save paths
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

SIMPLE_CNN_PATH = os.path.join(MODEL_DIR, "simple_cnn.h5")
DEEP_CNN_PATH = os.path.join(MODEL_DIR, "deep_cnn.h5")
MOBILENET_PATH = os.path.join(MODEL_DIR, "mobilenet.h5")

# Results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
EVALUATION_FILE = os.path.join(RESULTS_DIR, "evaluation_results.json")

# MODEL SETTINGS
IMG_SIZE = (32, 32)
NUM_CLASSES = 43
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# CLASS NAMES 
CLASS_NAMES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing',
    10: 'No passing for vehicles over 3.5 tons',
    11: 'Right-of-way at intersection', 12: 'Priority road',
    13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Vehicles over 3.5 tons prohibited', 17: 'No entry',
    18: 'General caution', 19: 'Dangerous curve left',
    20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road',
    24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing',
    30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 tons'
}