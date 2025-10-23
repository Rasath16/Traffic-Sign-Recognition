"""
Training script for traffic sign classification
- Trains Custom CNN and/or MobileNet
- Evaluates with accuracy and confusion matrix
- Includes data augmentation (bonus)
"""
import numpy as np
import os
import time
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from data_loader import load_gtsrb_dataset
from model import create_cnn_model, create_mobilenet_model
import config


def create_data_augmentation():
    """
    Create ImageDataGenerator for data augmentation (BONUS TASK)
    
    Augmentation techniques:
    - Rotation: ±15 degrees
    - Width/Height shift: ±10%
    - Zoom: 90-110%
    - Brightness: 70-130%
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.7, 1.3],
        horizontal_flip=False,  # Traffic signs shouldn't be flipped
        fill_mode='nearest'
    )
    return datagen


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(16, 14))
    
    # Plot heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray')
    
    plt.title('Confusion Matrix - Traffic Sign Classification\n', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix plot saved: {save_path}")
    plt.close()


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history['loss'], label='Train', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved: {save_path}")
    plt.close()


def train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                model_type='cnn', use_augmentation=True):
    """
    Train traffic sign classification model
    
    Args:
        model_type: 'cnn' or 'mobilenet'
        use_augmentation: Enable data augmentation (BONUS)
    
    Returns:
        model, history, metrics
    """
    
    print("\n" + "=" * 70)
    print(f"TRAINING {model_type.upper()} MODEL")
    print(f"Augmentation: {'ENABLED ✓' if use_augmentation else 'DISABLED ✗'}")
    print("=" * 70 + "\n")
    
    # Create model
    if model_type == 'cnn':
        model = create_cnn_model(
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
            num_classes=config.NUM_CLASSES
        )
    elif model_type == 'mobilenet':
        model = create_mobilenet_model(
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
            num_classes=config.NUM_CLASSES
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print(model.summary())
    print()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.CHECKPOINT_DIR, f'best_{model_type}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    if use_augmentation:
        # Train with data augmentation
        datagen = create_data_augmentation()
        datagen.fit(X_train)
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
            steps_per_epoch=len(X_train) // config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Train without augmentation
        history = model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    training_time = (time.time() - start_time) / 60  # Convert to minutes
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70 + "\n")
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Generate predictions
    print("\nGenerating predictions for confusion matrix...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # REQUIRED: Confusion Matrix
    print("\nComputing confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Save confusion matrix data
    cm_path = os.path.join(config.MODEL_DIR, f'confusion_matrix_{model_type}.npy')
    np.save(cm_path, cm)
    print(f"Confusion matrix saved: {cm_path}")
    
    # Plot confusion matrix
    cm_plot_path = os.path.join(config.MODEL_DIR, f'confusion_matrix_{model_type}.png')
    plot_confusion_matrix(cm, cm_plot_path)
    
    # Classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    
    # Per-class accuracy
    class_accuracy = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    
    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)
    print(f"Mean class accuracy: {np.mean(class_accuracy) * 100:.2f}%")
    print(f"\nBest performing classes:")
    best_indices = np.argsort(class_accuracy)[-5:][::-1]
    for idx in best_indices:
        print(f"  Class {idx:2d} ({config.CLASS_NAMES[idx][:40]}): {class_accuracy[idx]*100:.2f}%")
    
    print(f"\nWorst performing classes:")
    worst_indices = np.argsort(class_accuracy)[:5]
    for idx in worst_indices:
        print(f"  Class {idx:2d} ({config.CLASS_NAMES[idx][:40]}): {class_accuracy[idx]*100:.2f}%")
    
    # Save metrics
    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'train_accuracy': float(history.history['accuracy'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1]),
        'training_time_minutes': float(training_time),
        'class_accuracy': class_accuracy.tolist(),
    }
    
    metrics_path = os.path.join(config.MODEL_DIR, f'metrics_{model_type}.npy')
    np.save(metrics_path, metrics)
    print(f"\nMetrics saved: {metrics_path}")
    
    # Save training history
    history_path = os.path.join(config.MODEL_DIR, f'history_{model_type}.npy')
    np.save(history_path, history.history)
    print(f"Training history saved: {history_path}")
    
    # Plot training history
    history_plot_path = os.path.join(config.MODEL_DIR, f'training_history_{model_type}.png')
    plot_training_history(history.history, history_plot_path)
    
    # Save final model
    model_path = os.path.join(config.MODEL_DIR, f'traffic_sign_{model_type}.h5')
    model.save(model_path)
    print(f"Final model saved: {model_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Model:              {model_type.upper()}")
    print(f"Total training time: {training_time:.2f} minutes")
    print(f"Final train acc:    {metrics['train_accuracy']*100:.2f}%")
    print(f"Final val acc:      {metrics['val_accuracy']*100:.2f}%")
    print(f"Test accuracy:      {metrics['test_accuracy']*100:.2f}%")
    print(f"Data augmentation:  {'YES' if use_augmentation else 'NO'}")
    print("=" * 70 + "\n")
    
    return model, history, metrics


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Train Traffic Sign Recognition Model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='cnn',
        choices=['cnn', 'mobilenet', 'both'],
        help='Model type to train (default: cnn)'
    )
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print("\n" + "=" * 70)
    print("TRAFFIC SIGN RECOGNITION - TRAINING")
    print("=" * 70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_gtsrb_dataset()
    
    # Train model(s)
    use_augmentation = not args.no_augment
    
    if args.model == 'both':
        print("\n" + "=" * 70)
        print("TRAINING BOTH MODELS FOR COMPARISON (BONUS TASK)")
        print("=" * 70 + "\n")
        
        # Train Custom CNN
        print("\n>>> Training Custom CNN...")
        train_model(
            X_train, y_train, X_val, y_val, X_test, y_test,
            model_type='cnn',
            use_augmentation=use_augmentation
        )
        
        # Train MobileNet
        print("\n>>> Training MobileNet...")
        train_model(
            X_train, y_train, X_val, y_val, X_test, y_test,
            model_type='mobilenet',
            use_augmentation=use_augmentation
        )
        
        print("\n" + "=" * 70)
        print("BOTH MODELS TRAINED SUCCESSFULLY!")
        print("Check the models/saved_models/ folder for results")
        print("=" * 70)
        
    else:
        # Train single model
        train_model(
            X_train, y_train, X_val, y_val, X_test, y_test,
            model_type=args.model,
            use_augmentation=use_augmentation
        )
        
        print(f"\n{args.model.upper()} model trained successfully!")
        print("Check the models/saved_models/ folder for results")


if __name__ == "__main__":
    main()