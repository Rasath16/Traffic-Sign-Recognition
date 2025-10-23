"""
Utility functions for prediction and visualization
This script has been modified to include CLASS_NAMES 
to bypass persistent import issues with 'config.py'.
"""
import cv2
import numpy as np
from tensorflow import keras
# Removed 'import config'

# --- In-script definition of CLASS_NAMES for robustness ---
# This is a copy of the essential part of config.py for prediction utility
CLASS_NAMES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons', 
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield', 
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited', 
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left', 
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road', 
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work', 
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End of all speed and passing limits', 33: 'Turn right ahead', 
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right', 
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left', 
    40: 'Roundabout mandatory', 41: 'End of no passing', 
    42: 'End of no passing by vehicles over 3.5 metric tons'
}
TARGET_SIZE = (32, 32) # Assuming the model was trained on 32x32 images

def load_model(model_path):
    """Load trained model"""
    return keras.models.load_model(model_path)


def preprocess_single_image(image):
    """
    Preprocess a single image for prediction
    
    Args:
        image: Input image (numpy array or PIL Image)
    
    Returns:
        Preprocessed image ready for model
    """
    # Convert PIL Image to numpy if needed
    if hasattr(image, 'convert'):
        image = np.array(image)
    
    # Ensure RGB format
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize
    image_resized = cv2.resize(image, TARGET_SIZE)
    
    # Normalize
    image_normalized = image_resized.astype('float32') / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch


def predict_sign(model, image):
    """
    Predict traffic sign class
    
    Args:
        model: Trained Keras model
        image: Input image
    
    Returns:
        class_id: Predicted class ID
        confidence: Prediction confidence
        all_predictions: All class probabilities
    """
    # Preprocess image
    processed_image = preprocess_single_image(image)
    
    # Predict
    predictions = model.predict(processed_image, verbose=0)
    
    # Get class with highest probability
    class_id = np.argmax(predictions[0])
    confidence = predictions[0][class_id]
    
    return int(class_id), float(confidence), predictions[0]


def get_class_name(class_id):
    """Get class name from class ID"""
    # Use the in-script defined CLASS_NAMES
    return CLASS_NAMES.get(class_id, f"Unknown class {class_id}")