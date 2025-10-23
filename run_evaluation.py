#!/usr/bin/env python
"""
Standalone script to run model evaluation
Can be run directly or called from Streamlit app
"""

import os
import sys

# Add src to path
sys.path.append('src')

def main():
    """Run evaluation for all available models"""
    
    print("\n" + "="*60)
    print("TRAFFIC SIGN RECOGNITION - MODEL EVALUATION")
    print("="*60)
    
    # Check if models exist
    custom_cnn_exists = os.path.exists('models/custom_cnn_best.h5')
    mobilenet_exists = os.path.exists('models/mobilenet_best.h5')
    
    if not custom_cnn_exists and not mobilenet_exists:
        print("\n❌ ERROR: No trained models found!")
        print("Please train models first using: python src/train.py")
        return 1
    
    try:
        from evaluate import ModelEvaluator
        import numpy as np
        
        # Load test data once
        print("\n" + "="*60)
        print("LOADING TEST DATA")
        print("="*60)
        
        if custom_cnn_exists:
            evaluator = ModelEvaluator('models/custom_cnn_best.h5')
        else:
            evaluator = ModelEvaluator('models/mobilenet_best.h5')
        
        X_test, y_test = evaluator.load_test_data()
        
        # Evaluate Custom CNN
        if custom_cnn_exists:
            print("\n" + "="*60)
            print("EVALUATING CUSTOM CNN")
            print("="*60)
            
            evaluator_custom = ModelEvaluator('models/custom_cnn_best.h5')
            y_pred_custom, metrics_custom = evaluator_custom.evaluate(
                X_test, y_test, save_metrics=True, model_name='custom_cnn'
            )
            
            print("\nGenerating visualizations...")
            evaluator_custom.plot_confusion_matrix(
                y_test, y_pred_custom, 
                'models/custom_cnn_confusion_matrix.png'
            )
            evaluator_custom.plot_per_class_accuracy(
                y_test, y_pred_custom,
                'models/custom_cnn_per_class.png'
            )
            print("✅ Custom CNN evaluation complete!")
        
        # Evaluate MobileNet
        if mobilenet_exists:
            print("\n" + "="*60)
            print("EVALUATING MOBILENET V2")
            print("="*60)
            
            evaluator_mobilenet = ModelEvaluator('models/mobilenet_best.h5')
            y_pred_mobilenet, metrics_mobilenet = evaluator_mobilenet.evaluate(
                X_test, y_test, save_metrics=True, model_name='mobilenet_v2'
            )
            
            print("\nGenerating visualizations...")
            evaluator_mobilenet.plot_confusion_matrix(
                y_test, y_pred_mobilenet,
                'models/mobilenet_v2_confusion_matrix.png'
            )
            evaluator_mobilenet.plot_per_class_accuracy(
                y_test, y_pred_mobilenet,
                'models/mobilenet_v2_per_class.png'
            )
            print("✅ MobileNet V2 evaluation complete!")
        
        # Compare models if both exist
        if custom_cnn_exists and mobilenet_exists:
            print("\n" + "="*60)
            print("COMPARING MODELS")
            print("="*60)
            
            model_paths = [
                'models/custom_cnn_best.h5',
                'models/mobilenet_best.h5'
            ]
            model_names = ['Custom CNN', 'MobileNet V2']
            
            comparison = evaluator_custom.compare_models(
                model_paths, model_names, X_test, y_test
            )
            
            print("\n📊 Comparison Summary:")
            print("-" * 60)
            for name, metrics in comparison.items():
                print(f"\n{name}:")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            
            # Determine best model
            best_model = max(comparison.items(), key=lambda x: x[1]['accuracy'])
            print("\n" + "="*60)
            print(f"🏆 BEST MODEL: {best_model[0]}")
            print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
            print("="*60)
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\n📁 Generated files:")
        print("   - Metrics CSV files in models/")
        print("   - Confusion matrices in models/")
        print("   - Per-class accuracy plots in models/")
        print("   - Model comparison plot in models/")
        print("\n🚀 Run 'streamlit run app.py' to view results in the web interface")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR DURING EVALUATION")
        print("="*60)
        print(f"\n{str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)