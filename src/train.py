import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_loader import GTSRBDataLoader
from preprocessor import ImagePreprocessor
from model import TrafficSignModel

class ModelTrainer:
    """Handles model training pipeline"""
    
    def __init__(self, data_dir='data', model_dir='models'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.loader = GTSRBDataLoader(data_dir)
        self.preprocessor = ImagePreprocessor()
        self.model_builder = TrafficSignModel()
    
    def prepare_data(self, augment=True, augmentation_factor=2):
        """
        Load and prepare data for training
        
        Args:
            augment: whether to apply data augmentation
            augmentation_factor: augmentation multiplier
        
        Returns:
            X_train, X_val, y_train, y_val
        """
        print("="*50)
        print("LOADING DATA")
        print("="*50)
        
        # Load training data
        X_train, y_train = self.loader.load_data('Train')
        
        print("\n" + "="*50)
        print("PREPROCESSING")
        print("="*50)
        
        # Preprocess
        X_train = self.preprocessor.preprocess(X_train, apply_clahe=False)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Data augmentation (bonus requirement)
        if augment:
            print("\n" + "="*50)
            print("DATA AUGMENTATION")
            print("="*50)
            X_train, y_train = self.preprocessor.augment_data(
                X_train, y_train, augmentation_factor
            )
        
        # Convert labels to one-hot encoding
        y_train = self.preprocessor.prepare_labels(y_train)
        y_val = self.preprocessor.prepare_labels(y_val)
        
        print(f"\nFinal dataset sizes:")
        print(f"Training: {X_train.shape}")
        print(f"Validation: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def train_custom_model(self, X_train, X_val, y_train, y_val, 
                          epochs=50, batch_size=64):
        """
        Train custom CNN model
        
        Returns:
            trained model, history
        """
        print("\n" + "="*50)
        print("TRAINING CUSTOM CNN")
        print("="*50)
        
        # Build and compile model
        model = self.model_builder.build_custom_cnn()
        model = self.model_builder.compile_model(model, learning_rate=0.001)
        
        print("\nModel Summary:")
        model.summary()
        
        # Get callbacks
        callbacks = self.model_builder.get_callbacks(
            os.path.join(self.model_dir, 'custom_cnn_best.h5')
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(os.path.join(self.model_dir, 'custom_cnn_final.h5'))
        print(f"\nModel saved to {self.model_dir}")
        
        return model, history
    
    def train_mobilenet_model(self, X_train, X_val, y_train, y_val,
                             epochs=30, batch_size=64):
        """
        Train MobileNetV2 model (bonus requirement)
        
        Returns:
            trained model, history
        """
        print("\n" + "="*50)
        print("TRAINING MOBILENET V2")
        print("="*50)
        
        # Build and compile model
        model = self.model_builder.build_mobilenet()
        model = self.model_builder.compile_model(model, learning_rate=0.001)
        
        print("\nModel Summary:")
        model.summary()
        
        # Get callbacks
        callbacks = self.model_builder.get_callbacks(
            os.path.join(self.model_dir, 'mobilenet_best.h5')
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(os.path.join(self.model_dir, 'mobilenet_final.h5'))
        print(f"\nModel saved to {self.model_dir}")
        
        return model, history
    
    def plot_training_history(self, history, model_name='Custom CNN'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'{model_name.lower().replace(" ", "_")}_history.png'))
        plt.show()

def main():
    """Main training function"""
    # Initialize trainer
    trainer = ModelTrainer(data_dir='data', model_dir='models')
    
    # Prepare data
    X_train, X_val, y_train, y_val = trainer.prepare_data(
        augment=True, 
        augmentation_factor=2
    )
    
    # Train custom CNN
    custom_model, custom_history = trainer.train_custom_model(
        X_train, X_val, y_train, y_val,
        epochs=50,
        batch_size=64
    )
    trainer.plot_training_history(custom_history, 'Custom CNN')
    
    # Train MobileNetV2 (bonus)
    mobilenet_model, mobilenet_history = trainer.train_mobilenet_model(
        X_train, X_val, y_train, y_val,
        epochs=30,
        batch_size=64
    )
    trainer.plot_training_history(mobilenet_history, 'MobileNet V2')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()