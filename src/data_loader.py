import pandas as pd
import numpy as np
# from PIL import Image # REMOVED
import cv2 # ADDED
import os
from sklearn.model_selection import train_test_split
import config as config

def load_images_from_csv(csv_path, data_dir):
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    # Target image size
    img_height, img_width = config.IMG_SIZE
    
    print(f"Loading {len(df)} images...")
    for idx, row in df.iterrows():
        img_path = os.path.join(data_dir, row['Path'])
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                 continue
                 
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_resized = cv2.resize(img_rgb, config.IMG_SIZE)
            
            images.append(img_resized)
            labels.append(row['ClassId'])
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
        
        if (idx + 1) % 5000 == 0:
            print(f"  Loaded {idx + 1}/{len(df)} images")
    
    return np.array(images), np.array(labels)

def load_and_preprocess_data():
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

def preprocess_single_image(image_file):

    image_file.seek(0)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Could not decode image from uploaded file.")
        
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    img_resized = cv2.resize(img_rgb, config.IMG_SIZE)
    
    img_array = img_resized.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array