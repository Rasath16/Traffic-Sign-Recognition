"""
Training Script for Traffic Sign Recognition
Trains the model and saves the best version
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_preprocessing import DataPreprocessor
from model import TrafficSignModel


class ModelTrainer:
    def __init__(self, model, model_name='traffic_sign_model'):
        """
        Initialize trainer
        model: compiled Keras model
        model_name: name for saving the model
        """
        self.model = model
        self.model_name = model_name
        self.history = None
        
    def get_callbacks(self):
        """
        Setup training callbacks:
        - ModelCheckpoint: Save best model
        - EarlyStopping: Stop if no improvement
        - ReduceLROnPlateau: Reduce learning rate when stuck
        """
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        callbacks = [
            # Save the best model based on validation accuracy
            ModelCheckpoint(
                f'models/{self.model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Stop training if validation loss doesn't improve for 5 epochs
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate if stuck
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Reduce by half
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def get_data_augmentation(self):
        """
        Create data augmentation to artificially increase dataset size
        This helps prevent overfitting
        """
        datagen = ImageDataGenerator(
            rotation_range=15,      # Randomly rotate images by 15 degrees
            width_shift_range=0.1,  # Shift images horizontally
            height_shift_range=0.1, # Shift images vertically
            zoom_range=0.1,         # Zoom in/out
            brightness_range=[0.8, 1.2],  # Adjust brightness
            fill_mode='nearest'     # Fill empty pixels
        )
        
        return datagen
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=30, batch_size=32, use_augmentation=True):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            use_augmentation: Whether to use data augmentation
        """
        print("\n" + "="*60)
        print(f"TRAINING {self.model_name.upper()}")
        print("="*60)
        
        callbacks = self.get_callbacks()
        
        if use_augmentation:
            print("Using data augmentation...")
            datagen = self.get_data_augmentation()
            
            # Fit augmentation on training data
            datagen.fit(X_train)
            
            # Train with augmented data
            self.history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without augmentation
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        print(f"\nTraining completed!")
        print(f"Best model saved to: models/{self.model_name}_best.h5")
        
        return self.history
    
    def plot_training_history(self, save_path='models'):
        """
        Plot training and validation accuracy/loss
        """
        if self.history is None:
            print("No training history available!")
            return
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_path, f'{self.model_name}_training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {plot_path}")
        
        plt.show()


# Main training script
if __name__ == "__main__":
    print("="*60)
    print("TRAFFIC SIGN RECOGNITION - TRAINING")
    print("="*60)
    
    # 1. Load and preprocess data
    print("\n1. Loading data...")
    preprocessor = DataPreprocessor(img_size=32)
    X, y = preprocessor.load_data()
    X_train, X_val, y_train, y_val = preprocessor.split_data(X, y)
    
    # 2. Build model (choose one)
    print("\n2. Building model...")
    model_builder = TrafficSignModel(num_classes=43, img_size=32)
    
    # Option 1: Simple CNN (faster training, good for testing)
    # model = model_builder.build_simple_cnn()
    # model_name = 'simple_cnn'
    
    # Option 2: Deeper CNN (better accuracy)
    model = model_builder.build_deeper_cnn()
    model_name = 'deep_cnn'
    
    # Option 3: Transfer Learning (best accuracy, slower training)
    # model = model_builder.build_transfer_learning_model()
    # model_name = 'mobilenet_transfer'
    
    # Compile model
    model = model_builder.compile_model(model, learning_rate=0.001)
    model_builder.get_model_summary(model)
    
    # 3. Train model
    print("\n3. Training model...")
    trainer = ModelTrainer(model, model_name=model_name)
    history = trainer.train(
        X_train, y_train, 
        X_val, y_val,
        epochs=30,
        batch_size=32,
        use_augmentation=True
    )
    
    # 4. Plot training history
    print("\n4. Plotting training history...")
    trainer.plot_training_history()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)