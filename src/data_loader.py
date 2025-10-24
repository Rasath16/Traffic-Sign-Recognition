import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import config as config

def load_images_from_csv(csv_path, data_dir):
    """Load images from CSV file"""
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    print(f"Loading {len(df)} images...")
    for idx, row in df.iterrows():
        img_path = os.path.join(data_dir, row['Path'])
        try:
            img = Image.open(img_path)
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize(config.IMG_SIZE)
            images.append(np.array(img))
            labels.append(row['ClassId'])
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
        
        if (idx + 1) % 5000 == 0:
            print(f"  Loaded {idx + 1}/{len(df)} images")
    
    return np.array(images), np.array(labels)

def load_and_preprocess_data():
    """
    Main function to load all data
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\n" + "="*50)
    print("📂 LOADING DATA")
    print("="*50)
    
    # Load training data
    print("\n1️⃣ Loading Training Data...")
    X_train_full, y_train_full = load_images_from_csv(
        config.TRAIN_CSV, 
        config.DATA_DIR
    )
    
    # Load test data
    print("\n2️⃣ Loading Test Data...")
    X_test, y_test = load_images_from_csv(
        config.TEST_CSV,
        config.DATA_DIR
    )
    
    # Split training data into train and validation
    print("\n3️⃣ Splitting into Train/Validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=y_train_full
    )
    
    # Normalize images to [0, 1]
    print("\n4️⃣ Normalizing images...")
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    print("\n✅ DATA LOADING COMPLETE!")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    print("="*50 + "\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_single_image(image_path):
    """
    Preprocess a single image for prediction
    Handles RGBA, grayscale, and other formats
    """
    img = Image.open(image_path)
    
    # Convert to RGB if needed (fixes RGBA, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize(config.IMG_SIZE)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array