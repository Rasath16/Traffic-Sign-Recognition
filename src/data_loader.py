import pandas as pd
import numpy as np
import os
from PIL import Image
import cv2

class GTSRBDataLoader:
    """Handles loading of GTSRB dataset"""
    TARGET_SIZE = (32, 32)
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.class_names = self._load_class_names()
    
    def _load_class_names(self):
        """Load class names from Meta folder"""
        # Default class names mapping
        default_names = {
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
        
        try:
            meta_path = os.path.join(self.data_dir, 'Meta', 'signnames.csv')
            if os.path.exists(meta_path):
                df = pd.read_csv(meta_path)
                return dict(zip(df['ClassId'], df['SignName']))
            else:
                # Create Meta directory and save default names
                os.makedirs(os.path.join(self.data_dir, 'Meta'), exist_ok=True)
                # Save default names to CSV
                df = pd.DataFrame(list(default_names.items()), columns=['ClassId', 'SignName'])
                df.to_csv(meta_path, index=False)
                print(f"Created signnames.csv at {meta_path}")
                return default_names
        except Exception as e:
            print(f"Warning: Could not load class names: {e}")
            return default_names
    
    def load_data(self, split='Train'):
        """
        Load images and labels from Train or Test folder
        
        Args:
            split: 'Train' or 'Test'
        
        Returns:
            images: numpy array of images
            labels: numpy array of labels
        """
        csv_path = os.path.join(self.data_dir, f'{split}.csv')
        
        if not os.path.exists(csv_path):
            # The FileNotFoundError will stop execution before the ModelEvaluator tries to load data
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        images = []
        labels = []
        
        print(f"Loading {split} data...")
        for idx, row in df.iterrows():
            # row['Path'] in Train.csv is relative to the data_dir (e.g., 'Train/0/00000.ppm')
            img_path = os.path.join(self.data_dir, row['Path'])
            
            if os.path.exists(img_path):
                # Use cv2.IMREAD_COLOR to ensure 3 channels
                img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
                
                if img is not None:
                    # 1. Convert BGR to RGB (OpenCV default is BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 2. **FIX: Resize the image to a fixed size**
                    # This ensures all images have the same shape for np.array() conversion
                    img_resized = cv2.resize(img, self.TARGET_SIZE)
                    
                    images.append(img_resized) 
                    labels.append(row['ClassId'])
                
            
            if (idx + 1) % 5000 == 0:
                print(f"Loaded {len(images)}/{len(df)} images")
        
        # --- FIX: Explicitly return the arrays ---
        
        if len(images) == 0:
            print(f"WARNING: No images were loaded from {csv_path}. Returning empty arrays.")
            return np.array([]), np.array([])
            
        # This conversion now succeeds because all images in the list have the same size.
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels # <-- Explicit return of the expected tuple