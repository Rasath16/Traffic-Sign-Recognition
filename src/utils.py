"""
Utility functions for robust predictions
"""
import numpy as np
from PIL import Image, ImageEnhance
import config

def predict_with_tta(model, image_path, n_augmentations=5):
    """
    Test-Time Augmentation (TTA): Make predictions on multiple 
    variations of the image and average results for better accuracy
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    predictions_list = []
    
    # Helper function to preprocess
    def preprocess_pil_image(pil_img):
        pil_img = pil_img.resize(config.IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(pil_img).astype('float32') / 255.0
        return np.expand_dims(img_array, axis=0)
    
    # 1. Original image
    predictions_list.append(model.predict(preprocess_pil_image(img), verbose=0)[0])
    
    # 2. Brightness variations
    for brightness_factor in [0.7, 0.9, 1.1, 1.3]:
        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(brightness_factor)
        predictions_list.append(model.predict(preprocess_pil_image(img_bright), verbose=0)[0])
    
    # 3. Contrast variations
    for contrast_factor in [0.8, 1.2]:
        enhancer = ImageEnhance.Contrast(img)
        img_contrast = enhancer.enhance(contrast_factor)
        predictions_list.append(model.predict(preprocess_pil_image(img_contrast), verbose=0)[0])
    
    # Average all predictions
    final_prediction = np.mean(predictions_list, axis=0)
    
    return final_prediction