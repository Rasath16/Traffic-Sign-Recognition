"""
Data loading and preprocessing for GTSRB dataset
Handles Train.csv, Test.csv and image folders
"""
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import config


def load_images_from_csv(csv_path, base_dir):
    """
    Load images using CSV file
    
    Args:
        csv_path: Path to CSV file (Train.csv or Test.csv)
        base_dir: Base directory for images (should be data/raw/)
    
    Returns:
        images: numpy array of images
        labels: numpy array of labels
    """
    print(f"\nLoading data from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")
    print(f"CSV columns: {df.columns.tolist()}")
    
    images = []
    labels = []
    
    # Load each image
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        try:
            # Get path from CSV (e.g., "Train/20/00020_00000_00000.png")
            img_relative_path = str(row['Path'])
            
            # Construct full path: data/raw/ + Train/20/00020_00000_00000.png
            img_path = os.path.join(base_dir, img_relative_path)
            
            # Read image
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    target_size = (config.IMG_WIDTH, config.IMG_HEIGHT)
                    img_resized = cv2.resize(img, target_size)
                    images.append(img_resized)
                    labels.append(int(row['ClassId']))
                    successful += 1
                else:
                    failed += 1
                    if failed <= 3:
                        print(f"Warning: Could not read image: {img_path}")
            else:
                failed += 1
                if failed <= 3:
                    print(f"Warning: Image not found: {img_path}")
        
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"Error loading image at index {idx}: {e}")
    
    print(f"\nSuccessfully loaded: {successful} images")
    if failed > 0:
        print(f"Failed to load: {failed} images")
    
    if successful == 0:
        print("\n❌ ERROR: No images were loaded!")
        print("Please check:")
        print("1. CSV file paths are correct")
        print("2. Image files exist in the folders")
        print("3. Path format in CSV matches actual folder structure")
    
    return np.array(images), np.array(labels)


def load_images_from_folders(base_dir):
    """
    Load images directly from class folders (backup method)
    Assumes structure: base_dir/0/, base_dir/1/, ..., base_dir/42/
    
    Args:
        base_dir: Base directory containing class folders
    
    Returns:
        images: numpy array of images
        labels: numpy array of labels
    """
    print(f"\nLoading images from folder structure: {base_dir}")
    
    images = []
    labels = []
    
    # Get class folders (0-42)
    class_folders = sorted([f for f in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, f)) and f.isdigit()])
    
    if not class_folders:
        print(f"ERROR: No class folders found in {base_dir}")
        return np.array([]), np.array([])
    
    print(f"Found {len(class_folders)} class folders")
    
    for class_id in tqdm(class_folders, desc="Loading classes"):
        class_path = os.path.join(base_dir, class_id)
        
        # Get all images in this class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg'))]
        
        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(int(class_id))
    
    print(f"Successfully loaded {len(images)} images from {len(class_folders)} classes")
    
    return np.array(images), np.array(labels)


def preprocess_images(images, target_size=(32, 32)):
    """
    Preprocess images: resize and normalize
    
    Args:
        images: Array of images
        target_size: Target size (height, width)
    
    Returns:
        Preprocessed images
    """
    if len(images) == 0:
        return np.array([])
    
    processed = []
    
    print(f"\nPreprocessing {len(images)} images to size {target_size}...")
    
    for img in tqdm(images, desc="Preprocessing"):
        # Resize to target size
        img_resized = cv2.resize(img, target_size)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized.astype('float32') / 255.0
        
        processed.append(img_normalized)
    
    return np.array(processed)


def load_gtsrb_dataset():
    """
    Load complete GTSRB dataset from your structure
    
    Returns:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
    """
    print("=" * 70)
    print("LOADING GTSRB DATASET")
    print("=" * 70)
    
    # Check if files/folders exist
    if not os.path.exists(config.TRAIN_DIR):
        raise FileNotFoundError(f"Train folder not found at: {config.TRAIN_DIR}")
    if not os.path.exists(config.TEST_DIR):
        raise FileNotFoundError(f"Test folder not found at: {config.TEST_DIR}")
    if not os.path.exists(config.TRAIN_CSV):
        raise FileNotFoundError(f"Train.csv not found at: {config.TRAIN_CSV}")
    if not os.path.exists(config.TEST_CSV):
        raise FileNotFoundError(f"Test.csv not found at: {config.TEST_CSV}")
    
    # Load training data
    print("\n[1/4] Loading Training Data...")
    # Pass RAW_DATA_DIR (data/raw/) not TRAIN_DIR
    # Because CSV paths are like "Train/20/image.png" relative to data/raw/
    X_train_raw, y_train_full = load_images_from_csv(
        config.TRAIN_CSV, 
        config.RAW_DATA_DIR  # Important: use RAW_DATA_DIR not TRAIN_DIR
    )
    
    if len(X_train_raw) == 0:
        raise ValueError("Failed to load training data! Check dataset structure.")
    
    # Load test data
    print("\n[2/4] Loading Test Data...")
    X_test_raw, y_test = load_images_from_csv(
        config.TEST_CSV,
        config.RAW_DATA_DIR  # Important: use RAW_DATA_DIR not TEST_DIR
    )
    
    # If test data fails, create test split from training data
    if len(X_test_raw) == 0:
        print("Warning: Could not load test data, will split from training data")
        X_train_raw, X_test_raw, y_train_full, y_test = train_test_split(
            X_train_raw, y_train_full,
            test_size=0.15,
            random_state=42,
            stratify=y_train_full
        )
    
    # Print class distribution
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)
    unique_train, counts_train = np.unique(y_train_full, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"Number of classes (train): {len(unique_train)}")
    print(f"Number of classes (test): {len(unique_test)}")
    print(f"Total training samples: {len(y_train_full)}")
    print(f"Total test samples: {len(y_test)}")
    print(f"Class range: {unique_train.min()} to {unique_train.max()}")
    
    # Show class distribution
    print("\nTraining samples per class:")
    for cls, count in zip(unique_train[:10], counts_train[:10]):
        print(f"  Class {cls}: {count} samples")
    if len(unique_train) > 10:
        print(f"  ... (showing first 10 of {len(unique_train)} classes)")
    
    # Split training data into train and validation
    print("\n[3/4] Creating Train/Validation Split...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, 
        y_train_full,
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=y_train_full
    )
    
    # Preprocess all images
    print("\n[4/4] Preprocessing Images...")
    print("=" * 70)
    
    X_train = preprocess_images(X_train_raw, (config.IMG_HEIGHT, config.IMG_WIDTH))
    X_val = preprocess_images(X_val_raw, (config.IMG_HEIGHT, config.IMG_WIDTH))
    X_test = preprocess_images(X_test_raw, (config.IMG_HEIGHT, config.IMG_WIDTH))
    
    # Convert labels to categorical (one-hot encoding)
    print("\nConverting labels to categorical...")
    y_train = to_categorical(y_train, config.NUM_CLASSES)
    y_val = to_categorical(y_val, config.NUM_CLASSES)
    y_test = to_categorical(y_test, config.NUM_CLASSES)
    
    # Final summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Training samples:   {X_train.shape[0]:>6}")
    print(f"Validation samples: {X_val.shape[0]:>6}")
    print(f"Test samples:       {X_test.shape[0]:>6}")
    print(f"Image shape:        {X_train.shape[1:]}")
    print(f"Number of classes:  {config.NUM_CLASSES}")
    print("=" * 70 + "\n")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # Test data loading
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_gtsrb_dataset()
        print("✅ Data loading successful!")
        print(f"\nSample statistics:")
        print(f"  Train shape: {X_train.shape}")
        print(f"  Train min: {X_train.min():.3f}, max: {X_train.max():.3f}")
        print(f"  Label shape: {y_train.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()