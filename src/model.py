"""
CNN model architectures for traffic sign classification
1. Custom CNN
2. MobileNet (Transfer Learning)
"""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import config


def create_cnn_model(input_shape=(32, 32, 3), num_classes=43):
    """
    Create custom CNN model
    
    Architecture:
    - 3 Convolutional blocks with batch normalization and dropout
    - Dense layers with regularization
    - ~500K parameters
    """
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='CustomCNN')
    
    return model


def create_mobilenet_model(input_shape=(32, 32, 3), num_classes=43):
    """
    Create MobileNet model with transfer learning
    
    Architecture:
    - Pre-trained MobileNetV2 base (frozen)
    - Custom classification head
    - ~2M parameters (frozen base)
    """
    
    # Input
    inputs = layers.Input(shape=input_shape)
    
    # Resize for MobileNet (requires at least 32x32, but works better with larger)
    # Upscale to 96x96 for better feature extraction
    x = layers.UpSampling2D(size=(3, 3))(inputs)  # 32x32 -> 96x96
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add base model
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='MobileNet_Transfer')
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Custom CNN...")
    cnn = create_cnn_model()
    cnn.summary()
    
    print("\n" + "="*70 + "\n")
    
    print("Testing MobileNet...")
    mobilenet = create_mobilenet_model()
    mobilenet.summary()