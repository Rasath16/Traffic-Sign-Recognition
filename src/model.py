"""
CNN Model Architecture for Traffic Sign Recognition
Two models: Simple CNN and Transfer Learning with MobileNet
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

class TrafficSignModel:
    def __init__(self, num_classes=43, img_size=32):
        """
        Initialize model builder
        num_classes: 43 traffic sign categories
        img_size: input image size (32x32)
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.input_shape = (img_size, img_size, 3)
    
    def build_simple_cnn(self):
        """
        Build a simple CNN from scratch
        
        Architecture:
        - 2 Convolutional blocks (Conv -> ReLU -> MaxPool)
        - Flatten
        - 2 Dense layers
        - Output layer with softmax
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),  # Prevent overfitting
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer (43 classes)
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_deeper_cnn(self):
        """
        Build a deeper CNN for better accuracy
        
        Architecture:
        - 3 Convolutional blocks
        - Batch Normalization for stable training
        - More filters for complex patterns
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_learning_model(self):
        """
        Build model using Transfer Learning with MobileNetV2
        Pre-trained on ImageNet, fine-tuned for traffic signs
        """
        # Load pre-trained MobileNetV2 (without top layer)
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers (we'll only train the top layers)
        base_model.trainable = False
        
        # Add custom classification layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile model with optimizer, loss, and metrics
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',  # For integer labels
            metrics=['accuracy']
        )
        
        return model
    
    def get_model_summary(self, model):
        """
        Print model architecture and parameters
        """
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        model.summary()
        
        # Count parameters
        total_params = model.count_params()
        print(f"\nTotal parameters: {total_params:,}")


# Example usage
if __name__ == "__main__":
    # Create model builder
    model_builder = TrafficSignModel(num_classes=43, img_size=32)
    
    # Build simple CNN
    print("Building Simple CNN...")
    simple_model = model_builder.build_simple_cnn()
    simple_model = model_builder.compile_model(simple_model)
    model_builder.get_model_summary(simple_model)
    
    # Build deeper CNN
    print("\n\nBuilding Deeper CNN...")
    deep_model = model_builder.build_deeper_cnn()
    deep_model = model_builder.compile_model(deep_model)
    model_builder.get_model_summary(deep_model)
    
    # Build transfer learning model
    print("\n\nBuilding Transfer Learning Model (MobileNetV2)...")
    transfer_model = model_builder.build_transfer_learning_model()
    transfer_model = model_builder.compile_model(transfer_model)
    model_builder.get_model_summary(transfer_model)