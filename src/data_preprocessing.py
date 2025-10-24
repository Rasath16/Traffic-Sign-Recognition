"""
Data Preprocessing for Traffic Sign Recognition
This file handles loading and preparing images for training
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd

class DataPreprocessor:
    def __init__(self, img_size=32):
        """
        Initialize preprocessor
        img_size: resize all images to this size (32x32 pixels)
        """
        self.img_size = img_size
        self.num_classes = 43  # GTSRB has 43 traffic sign classes
        
    def load_data(self, data_dir='data'):
        """
        Load training data from the Train folder
        Returns: X (images), y (labels)
        """
        print("Loading training data...")
        
        images = []
        labels = []
        
        # Loop through each class folder (0 to 42)
        train_path = os.path.join(data_dir, 'Train')
        
        for class_num in range(self.num_classes):
            class_path = os.path.join(train_path, str(class_num))
            
            # Get all images in this class folder
            image_files = os.listdir(class_path)
            
            for img_file in image_files:
                if img_file.endswith('.png') or img_file.endswith('.jpg'):
                    # Read image
                    img_path = os.path.join(class_path, img_file)
                    img = cv2.imread(img_path)
                    
                    # Preprocess image
                    img = self.preprocess_image(img)
                    
                    images.append(img)
                    labels.append(class_num)
            
            if (class_num + 1) % 10 == 0:
                print(f"Loaded {class_num + 1}/{self.num_classes} classes")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"\nDataset loaded: {X.shape[0]} images")
        print(f"Image shape: {X.shape[1:]}")
        
        return X, y
    
    def preprocess_image(self, img):
        """
        Preprocess a single image:
        1. Resize to 32x32
        2. Normalize pixel values to [0, 1]
        """
        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize pixel values from [0, 255] to [0, 1]
        img = img / 255.0
        
        return img
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and validation sets
        test_size: percentage of data to use for validation (20%)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Keep same class distribution
        )
        
        print(f"\nTraining samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        
        return X_train, X_val, y_train, y_val
    
    def load_test_data(self, data_dir='data'):
        """
        Load test data from Test.csv
        """
        print("\nLoading test data...")
        
        test_csv = os.path.join(data_dir, 'Test.csv')
        test_df = pd.read_csv(test_csv)
        
        images = []
        labels = []
        
        test_folder = os.path.join(data_dir, 'Test')
        
        for idx, row in test_df.iterrows():
            img_path = os.path.join(test_folder, row['Path'])
            img = cv2.imread(img_path)
            
            if img is not None:
                img = self.preprocess_image(img)
                images.append(img)
                labels.append(row['ClassId'])
        
        X_test = np.array(images)
        y_test = np.array(labels)
        
        print(f"Test samples: {X_test.shape[0]}")
        
        return X_test, y_test


# Example usage
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = DataPreprocessor(img_size=32)
    
    # Load training data
    X, y = preprocessor.load_data()
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = preprocessor.split_data(X, y)
    
    # Load test data
    X_test, y_test = preprocessor.load_test_data()