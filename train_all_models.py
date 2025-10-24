import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.data_loader import load_and_preprocess_data
from src.models import create_simple_cnn, create_deep_cnn, create_mobilenet
from src.train import train_model
from src.evaluate import evaluate_model, save_all_results
import config

def main():
    print("\n" + "="*60)
    print("🚦 TRAFFIC SIGN RECOGNITION - TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    
    # Step 2: Define models
    models_config = [
        ('Simple CNN', create_simple_cnn(), config.SIMPLE_CNN_PATH),
        ('Deep CNN', create_deep_cnn(), config.DEEP_CNN_PATH),
        ('MobileNet', create_mobilenet(), config.MOBILENET_PATH),
    ]
    
    # Step 3: Train all models
    all_results = {}
    
    for model_name, model, save_path in models_config:
        # Train
        history, training_time = train_model(
            model, model_name,
            X_train, y_train,
            X_val, y_val,
            save_path
        )
        
        # Evaluate
        results = evaluate_model(model, model_name, X_test, y_test)
        results['training_time'] = training_time
        results['final_train_acc'] = float(history.history['accuracy'][-1] * 100)
        results['final_val_acc'] = float(history.history['val_accuracy'][-1] * 100)
        
        all_results[model_name] = results
    
    # Step 4: Save results
    save_all_results(all_results)
    
    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED AND EVALUATED!")
    print("="*60)
    print("\n📊 SUMMARY:")
    for name, results in all_results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  F1-Score: {results['f1_score']:.2f}%")
        print(f"  Training Time: {results['training_time']:.2f}s")
    
    print("\n🎉 Now run: streamlit run app.py")

if __name__ == "__main__":
    main()