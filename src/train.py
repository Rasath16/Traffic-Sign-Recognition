from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import config as config

def train_model(model, model_name, X_train, y_train, X_val, y_val, save_path):

    print("\n" + "="*50)
    print(f"🚀 TRAINING: {model_name}")
    print("="*50)
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )
    
    # Train
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Save model
    model.save(save_path)
    print(f"\n✅ Model saved to: {save_path}")
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    print("="*50 + "\n")
    
    return history, training_time