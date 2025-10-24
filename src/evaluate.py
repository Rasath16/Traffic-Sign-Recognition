"""
Model Evaluation Script
Test the trained model and visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras
import cv2

from data_preprocessing import DataPreprocessor


class ModelEvaluator:
    def __init__(self, model_path):
        """
        Initialize evaluator
        model_path: path to saved model (.h5 file)
        """
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        self.class_names = self.get_class_names()
        
    def get_class_names(self):
        """
        Get traffic sign class names
        """
        # GTSRB class names (43 classes)
        classes = {
            0: 'Speed limit (20km/h)',
            1: 'Speed limit (30km/h)',
            2: 'Speed limit (50km/h)',
            3: 'Speed limit (60km/h)',
            4: 'Speed limit (70km/h)',
            5: 'Speed limit (80km/h)',
            6: 'End of speed limit (80km/h)',
            7: 'Speed limit (100km/h)',
            8: 'Speed limit (120km/h)',
            9: 'No passing',
            10: 'No passing veh over 3.5 tons',
            11: 'Right-of-way at intersection',
            12: 'Priority road',
            13: 'Yield',
            14: 'Stop',
            15: 'No vehicles',
            16: 'Veh > 3.5 tons prohibited',
            17: 'No entry',
            18: 'General caution',
            19: 'Dangerous curve left',
            20: 'Dangerous curve right',
            21: 'Double curve',
            22: 'Bumpy road',
            23: 'Slippery road',
            24: 'Road narrows on the right',
            25: 'Road work',
            26: 'Traffic signals',
            27: 'Pedestrians',
            28: 'Children crossing',
            29: 'Bicycles crossing',
            30: 'Beware of ice/snow',
            31: 'Wild animals crossing',
            32: 'End speed + passing limits',
            33: 'Turn right ahead',
            34: 'Turn left ahead',
            35: 'Ahead only',
            36: 'Go straight or right',
            37: 'Go straight or left',
            38: 'Keep right',
            39: 'Keep left',
            40: 'Roundabout mandatory',
            41: 'End of no passing',
            42: 'End no passing veh > 3.5 tons'
        }
        return classes
    
    def evaluate_on_test_set(self, X_test, y_test):
        """
        Evaluate model on test set
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL ON TEST SET")
        print("="*60)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                   target_names=[self.class_names[i] for i in range(43)],
                                   digits=3))
        
        return y_pred_classes, accuracy
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path='models/confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        print("\nGenerating confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=range(43), yticklabels=range(43))
        plt.title('Confusion Matrix - Traffic Sign Recognition', fontsize=16)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_sample_predictions(self, X_test, y_test, num_samples=15):
        """
        Plot sample predictions with true and predicted labels
        """
        # Get random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        # Make predictions
        predictions = self.model.predict(X_test[indices])
        pred_classes = np.argmax(predictions, axis=1)
        
        # Plot
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Display image
            img = X_test[idx]
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Get true and predicted labels
            true_label = y_test[idx]
            pred_label = pred_classes[i]
            confidence = predictions[i][pred_label] * 100
            
            # Color: green if correct, red if wrong
            color = 'green' if true_label == pred_label else 'red'
            
            # Title with labels
            title = f"True: {self.class_names[true_label][:20]}\n"
            title += f"Pred: {self.class_names[pred_label][:20]}\n"
            title += f"Conf: {confidence:.1f}%"
            
            axes[i].set_title(title, fontsize=8, color=color)
        
        plt.tight_layout()
        plt.savefig('models/sample_predictions.png', dpi=150, bbox_inches='tight')
        print(f"Sample predictions saved to: models/sample_predictions.png")
        plt.show()
    
    def predict_single_image(self, img_path):
        """
        Predict a single image
        """
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Predict
        prediction = self.model.predict(img)
        pred_class = np.argmax(prediction)
        confidence = prediction[0][pred_class] * 100
        
        print(f"\nPrediction: {self.class_names[pred_class]}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Display image
        display_img = cv2.imread(img_path)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(display_img)
        plt.title(f"Predicted: {self.class_names[pred_class]}\nConfidence: {confidence:.2f}%")
        plt.axis('off')
        plt.show()
        
        return pred_class, confidence
    
    def analyze_per_class_accuracy(self, y_test, y_pred):
        """
        Show accuracy for each class
        """
        print("\n" + "="*60)
        print("PER-CLASS ACCURACY")
        print("="*60)
        
        # Calculate accuracy for each class
        class_accuracies = {}
        
        for class_id in range(43):
            # Get indices where true label is this class
            class_indices = np.where(y_test == class_id)[0]
            
            if len(class_indices) > 0:
                # Calculate accuracy for this class
                class_predictions = y_pred[class_indices]
                correct = np.sum(class_predictions == class_id)
                accuracy = (correct / len(class_indices)) * 100
                
                class_accuracies[class_id] = accuracy
        
        # Sort by accuracy (worst to best)
        sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1])
        
        # Print worst 10 classes
        print("\nWorst 10 Classes:")
        for class_id, acc in sorted_classes[:10]:
            print(f"Class {class_id}: {self.class_names[class_id][:30]:30} - {acc:.2f}%")
        
        # Print best 10 classes
        print("\nBest 10 Classes:")
        for class_id, acc in sorted_classes[-10:]:
            print(f"Class {class_id}: {self.class_names[class_id][:30]:30} - {acc:.2f}%")


# Main evaluation script
if __name__ == "__main__":
    print("="*60)
    print("TRAFFIC SIGN RECOGNITION - EVALUATION")
    print("="*60)
    
    # 1. Load test data
    print("\n1. Loading test data...")
    preprocessor = DataPreprocessor(img_size=32)
    X_test, y_test = preprocessor.load_test_data()
    
    # 2. Load trained model
    print("\n2. Loading trained model...")
    model_path = 'models/deep_cnn_best.h5'  # Change this to your model name
    evaluator = ModelEvaluator(model_path)
    
    # 3. Evaluate on test set
    print("\n3. Evaluating model...")
    y_pred, accuracy = evaluator.evaluate_on_test_set(X_test, y_test)
    
    # 4. Plot confusion matrix
    print("\n4. Creating confusion matrix...")
    evaluator.plot_confusion_matrix(y_test, y_pred)
    
    # 5. Plot sample predictions
    print("\n5. Plotting sample predictions...")
    evaluator.plot_sample_predictions(X_test, y_test, num_samples=15)
    
    # 6. Analyze per-class accuracy
    print("\n6. Analyzing per-class accuracy...")
    evaluator.analyze_per_class_accuracy(y_test, y_pred)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)