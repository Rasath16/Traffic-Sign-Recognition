import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support
)
from tensorflow.keras.models import load_model
from data_loader import GTSRBDataLoader
from preprocessor import ImagePreprocessor

class ModelEvaluator:
    """Handles model evaluation and metrics"""
    
    def __init__(self, model_path, data_dir='data'):
        self.model = load_model(model_path)
        self.data_dir = data_dir
        self.loader = GTSRBDataLoader(data_dir)
        self.preprocessor = ImagePreprocessor()
    
    def load_test_data(self):
        """Load and preprocess test data"""
        print("Loading test data...")
        X_test, y_test = self.loader.load_data('Test')
        X_test = self.preprocessor.preprocess(X_test)
        return X_test, y_test
    
    def evaluate(self, X_test, y_test, save_metrics=True, model_name='model'):
        """
        Evaluate model on test data
        
        Returns:
            predictions, metrics dictionary
        """
        print("\nEvaluating model...")
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Save metrics to CSV
        if save_metrics:
            import pandas as pd
            metrics_df = pd.DataFrame([metrics])
            metrics_path = f'models/{model_name}_metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            print(f"\nMetrics saved to {metrics_path}")
        
        return y_pred, metrics
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=range(43), yticklabels=range(43))
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nConfusion matrix saved to {save_path}")
        plt.show()
        
        return cm
    
    def plot_per_class_accuracy(self, y_test, y_pred, save_path='per_class_accuracy.png'):
        """Plot per-class accuracy"""
        classes = np.unique(y_test)
        accuracies = []
        
        for cls in classes:
            mask = y_test == cls
            if mask.sum() > 0:
                acc = (y_pred[mask] == y_test[mask]).mean()
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        plt.figure(figsize=(16, 6))
        plt.bar(classes, accuracies, color='steelblue', alpha=0.8)
        plt.xlabel('Class ID', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=14)
        plt.xticks(classes)
        plt.ylim([0, 1.05])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Per-class accuracy plot saved to {save_path}")
        plt.show()
    
    def get_classification_report(self, y_test, y_pred):
        """Generate detailed classification report"""
        report = classification_report(
            y_test, y_pred,
            target_names=[self.loader.class_names.get(i, f'Class_{i}') 
                         for i in range(43)],
            output_dict=True
        )
        return report
    
    def compare_models(self, model_paths, model_names, X_test, y_test):
        """
        Compare multiple models (bonus requirement)
        
        Args:
            model_paths: list of model file paths
            model_names: list of model names
            X_test: test images
            y_test: test labels
        
        Returns:
            comparison dictionary
        """
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        comparison = {}
        
        for path, name in zip(model_paths, model_names):
            print(f"\nEvaluating {name}...")
            model = load_model(path)
            
            y_pred_probs = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            comparison[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Plot comparison
        self.plot_model_comparison(comparison)
        
        return comparison
    
    def plot_model_comparison(self, comparison, save_path='model_comparison.png'):
        """Plot model comparison"""
        models = list(comparison.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            values = [comparison[model][metric] for model in models]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison', fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nModel comparison plot saved to {save_path}")
        plt.show()

def main():
    """Main evaluation function"""
    import os
    
    # Evaluate custom CNN
    print("\n" + "="*50)
    print("EVALUATING CUSTOM CNN")
    print("="*50)
    
    evaluator_custom = ModelEvaluator('models/custom_cnn_best.h5')
    X_test, y_test = evaluator_custom.load_test_data()
    y_pred_custom, metrics_custom = evaluator_custom.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    evaluator_custom.plot_confusion_matrix(y_test, y_pred_custom, 
                                          'models/custom_cnn_confusion_matrix.png')
    
    # Plot per-class accuracy
    evaluator_custom.plot_per_class_accuracy(y_test, y_pred_custom,
                                            'models/custom_cnn_per_class.png')
    
    # Compare models (bonus requirement)
    if os.path.exists('models/mobilenet_best.h5'):
        print("\n" + "="*50)
        print("COMPARING MODELS")
        print("="*50)
        
        model_paths = [
            'models/custom_cnn_best.h5',
            'models/mobilenet_best.h5'
        ]
        model_names = ['Custom CNN', 'MobileNet V2']
        
        evaluator_custom.compare_models(model_paths, model_names, X_test, y_test)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()