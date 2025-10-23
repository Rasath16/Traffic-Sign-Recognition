import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImagePreprocessor:
    """Handles image preprocessing and augmentation"""
    
    def __init__(self, target_size=(32, 32), normalize=True):
        self.target_size = target_size
        self.normalize = normalize
    
    def resize_images(self, images):
        """Resize images to target size"""
        resized = []
        for img in images:
            img_resized = cv2.resize(img, self.target_size)
            resized.append(img_resized)
        return np.array(resized)
    
    def normalize_images(self, images):
        """Normalize pixel values to [0, 1]"""
        return images.astype('float32') / 255.0
    
    def apply_clahe(self, images):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = []
        
        for img in images:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            # Apply CLAHE to L channel
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            # Convert back to RGB
            enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            enhanced.append(enhanced_img)
        
        return np.array(enhanced)
    
    def preprocess(self, images, apply_clahe=False):
        """
        Complete preprocessing pipeline
        
        Args:
            images: numpy array of images
            apply_clahe: whether to apply CLAHE enhancement
        
        Returns:
            preprocessed images
        """
        # Resize
        images = self.resize_images(images)
        
        # Optional CLAHE
        if apply_clahe:
            images = self.apply_clahe(images)
        
        # Normalize
        if self.normalize:
            images = self.normalize_images(images)
        
        return images
    
    def create_augmentation_generator(self, augmentation_factor=5):
        """
        Create ImageDataGenerator for data augmentation
        
        Args:
            augmentation_factor: how many augmented versions per image
        
        Returns:
            ImageDataGenerator object
        """
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        return datagen
    
    def augment_data(self, X_train, y_train, augmentation_factor=2):
        """
        Augment training data
        
        Args:
            X_train: training images
            y_train: training labels
            augmentation_factor: multiplier for dataset size
        
        Returns:
            augmented X_train, y_train
        """
        datagen = self.create_augmentation_generator()
        
        augmented_images = []
        augmented_labels = []
        
        # Keep original data
        augmented_images.extend(X_train)
        augmented_labels.extend(y_train)
        
        print(f"Augmenting data with factor {augmentation_factor}...")
        
        # Generate augmented images
        for i in range(len(X_train)):
            img = X_train[i].reshape((1,) + X_train[i].shape)
            label = y_train[i]
            
            aug_iter = datagen.flow(img, batch_size=1)
            
            for _ in range(augmentation_factor - 1):
                aug_img = next(aug_iter)[0]
                augmented_images.append(aug_img)
                augmented_labels.append(label)
            
            if (i + 1) % 5000 == 0:
                print(f"Augmented {i + 1}/{len(X_train)} images")
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def prepare_labels(self, labels, num_classes=43):
        """Convert labels to one-hot encoding"""
        return to_categorical(labels, num_classes)