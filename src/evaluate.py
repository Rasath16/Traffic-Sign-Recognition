import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
import json
import time
import config

def evaluate_model(model, model_name, X_test, y_test):
    """
    Comprehensive model evaluation
    Returns dictionary with all metrics
    """
    print(f"\n📊 Evaluating {model_name}...")
    
    # Predictions
    start_time = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    inference_time = time.time() - start_time
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy * 100),
        'precision': float(precision * 100),
        'recall': float(recall * 100),
        'f1_score': float(f1 * 100),
        'inference_time': float(inference_time),
        'avg_inference_per_image': float(inference_time / len(X_test) * 1000),  # ms
        'total_params': int(model.count_params()),
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': {
            config.CLASS_NAMES[i]: float(acc * 100) 
            for i, acc in enumerate(class_accuracy)
        }
    }
    
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   F1-Score: {f1*100:.2f}%")
    print(f"   Inference time: {inference_time:.2f}s")
    
    return results

def save_all_results(results_dict):
    """Save evaluation results to JSON"""
    with open(config.EVALUATION_FILE, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"\n💾 Results saved to: {config.EVALUATION_FILE}")

def load_evaluation_results():
    """Load saved evaluation results"""
    try:
        with open(config.EVALUATION_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None